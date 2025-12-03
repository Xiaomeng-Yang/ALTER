# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
import torch.nn.functional as F
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os, random

from custom_unet.models import DiT_models
from hypernet import DiT_HyperStructure
from diffusion import create_diffusion
from utils.prune_utils import select_pruning_mask_dit
from utils.loss_utils import match_loss, router_balance_loss, add_block_hooks_dit
import wandb


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features_dir, labels_dir, flip=0):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))
        self.flip = flip

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))

        if self.flip>0:
            if random.random() < self.flip:
                features = features[1:]
            else:
                features = features[:1]
        return torch.from_numpy(features), torch.from_numpy(labels)


def dit_step(model, x, y, ema, opt, diffusion, pruning_mask, args,
             teacher_model=None, act_student=None, act_teacher=None):
    model.train()
    model.requires_grad_(True)
    device = x.device
    loss_dict = {}
    # Forward pass through the model
    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
    selected_mask = select_pruning_mask_dit(pruning_mask, t)

    model_term = diffusion.training_losses(model, x, t, model_kwargs=dict(y=y),
                                          pruning_mask=selected_mask)
    if args.distill and teacher_model is not None:
        # Distillation loss
        with torch.no_grad():
            teacher_term = diffusion.training_losses(teacher_model, x, t, model_kwargs=dict(y=y),
                                                      pruning_mask=selected_mask)
        distill_loss = F.mse_loss(model_term["output"].float(), teacher_term["output"].float(), reduction="mean")
        loss_dict['loss_dit_distill'] = distill_loss.item()

        feature_loss = torch.tensor(0.0, device=device)
        if args.feature_kd:
            for key in act_student.keys():
                feature_loss += F.mse_loss(act_student[key].float(), act_teacher[key].detach().float(), reduction="mean")
            feature_loss /= len(act_student.keys())
            loss_dict['loss_dit_block'] = feature_loss.item()

    if not args.distill_only:
        loss = model_term["loss"].mean()
        loss_dict['loss_dit_sd'] = loss.item()
        if args.distill:
            total_loss = loss + args.distill_rate * distill_loss + args.feature_kd_rate * feature_loss
        else:
            total_loss = loss
    else:
        total_loss = distill_loss + args.feature_kd_rate * feature_loss

    # backpropagate
    opt.zero_grad()
    total_loss.backward()
    opt.step()
    update_ema(ema, model.module)
    return total_loss, loss_dict


def hypernet_step(hypernet, model, x, y, opt_hyper, diffusion, args,
                  teacher_model=None, act_student=None, act_teacher=None):
    # a) freeze unet & unfreeze hypernet()
    model.requires_grad_(False)
    hypernet.train()
    hypernet.requires_grad_(True)
    loss_dict = {}
    device = x.device

    dummy_input = torch.tensor(0).to(device)
    pruning_mask, router_out, router_logits, _ = hypernet(dummy_input)
    # Forward pass through the hypernet
    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
    selected_mask = select_pruning_mask_dit(pruning_mask, t)
    model_term = diffusion.training_losses(model, x, t, model_kwargs=dict(y=y), pruning_mask=selected_mask)

    if args.distill and teacher_model is not None:
        # Distillation loss
        with torch.no_grad():
            teacher_term = diffusion.training_losses(teacher_model, x, t, model_kwargs=dict(y=y),
                                                      pruning_mask=selected_mask)
        distill_loss = F.mse_loss(model_term["output"].float(), teacher_term["output"].float(), reduction="mean")

        feature_loss = torch.tensor(0.0, device=device)
        if args.feature_kd:
            for key in act_student.keys():
                feature_loss += F.mse_loss(act_student[key].float(), act_teacher[key].detach().float(), reduction="mean")
            feature_loss /= len(act_student.keys())
    if not args.distill_only:
        loss = model_term["loss"].mean()
        loss_dict['loss_dit_sd'] = loss.item()
        if args.distill:
            dit_loss = loss + args.distill_rate * distill_loss + args.feature_kd_rate * feature_loss
        else:
            dit_loss = loss
    else:
        dit_loss = distill_loss + args.feature_kd_rate * feature_loss

    # d) mask constrain: total pruning ratio
    pruning_ratio = pruning_mask.sum() / pruning_mask.numel()
    ratio_loss = match_loss(pruning_ratio, args.pruning_ratio)
    loss_dict['loss_ratio'] = ratio_loss.item()

    hyper_loss = dit_loss + args.ratio_loss_rate * ratio_loss

    if args.router_balance:
        router_loss = router_balance_loss(router_logits, router_out, args.n_experts)
        loss_dict['loss_router_balance'] = router_loss.item()
        hyper_loss = hyper_loss + args.router_balance_rate * router_loss

    # Backward pass and optimization step
    opt_hyper.zero_grad()
    hyper_loss.backward()
    opt_hyper.step()

    hypernet.eval()  # Set hypernet back to eval mode
    with torch.no_grad():
        pruning_mask, router_out, _, _ = hypernet(dummy_input)

    return pruning_mask, hyper_loss, loss_dict


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    def print_rank_0(msg):
        if rank == 0:
            print(msg)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        print_rank_0(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    if rank == 0:
        if args.prefix is not None:
            wandb_resume = os.path.join(experiment_dir, args.prefix+'.wandb_id')
            if os.path.exists(wandb_resume):
                # read the wandb id from the txt file
                with open(wandb_resume, 'r') as f:
                    wandb_id = f.read()
            else:
                wandb_id = wandb.util.generate_id()
                with open(wandb_resume, 'w') as f:
                    f.write(wandb_id)
            wandb.init(project="DiTMoe", name=args.prefix, id=wandb_id, resume="allow", config=args)
        else:
            wandb.init(project="DiTMoe", name=args.prefix, config=args)

    # Create model:
    # pretrained DiT model initialization:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )

    initial_ckpt = torch.load(args.load_weight, map_location='cpu')
    if 'ema' in initial_ckpt:
        model.load_state_dict(initial_ckpt['ema'], strict=True)
        print_rank_0(f"Loaded initial EMA weights from {args.load_weight}")
    elif 'model' in initial_ckpt:
        model.load_state_dict(initial_ckpt['model'], strict=True)
        print_rank_0(f"Loaded initial weights from {args.load_weight}")
    else:
        model.load_state_dict(initial_ckpt, strict=True)
        print_rank_0(f"Loaded plain weights from {args.load_weight}")

    if args.distill:
        teacher_model = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes
        )
        teacher_ckpt = torch.load(args.load_weight, map_location='cpu')
        if 'ema' in teacher_ckpt:   
            teacher_model.load_state_dict(teacher_ckpt['ema'], strict=True)
            print_rank_0(f"Loaded teacher EMA weights from {args.load_weight}")
        elif 'model' in teacher_ckpt:
            teacher_model.load_state_dict(teacher_ckpt['model'], strict=True)
            print_rank_0(f"Loaded teacher weights from {args.load_weight}")
        else:
            teacher_model.load_state_dict(teacher_ckpt, strict=True)
            print_rank_0(f"Loaded plain teacher weights from {args.load_weight}")
    else:
        teacher_model = None

    # get the layers of the DiT model:
    p_structures = model.count_p_structures()
    # build controllernetwork for mask generation
    time_embedding = model.get_all_time_embeds(args.T)  # Get time embeddings from the model

    # initialize the hypernet
    hypernet = DiT_HyperStructure(time_embedding, args.n_experts, p_structures, T=0.4, base=4)
    hypernet.eval()
    with torch.no_grad():
        pruning_mask, router_out, _, _ = hypernet(torch.tensor(0))

    number_of_zeros = (pruning_mask == 0).sum().item()
    time_router = torch.argmax(router_out, dim=1).cpu().numpy()
    logger.info("Initialized timestep router: {}".format(time_router))
    logger.info(f"Initialized binary mask has {number_of_zeros} masked layers within the vector.")
    # logger.info("Random initialized Mask: {}".format(cur_maskVec))
    logger.info("=====> Mask ControllerNetwork Initialization Done. <=====")

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    teacher_model = DDP(teacher_model.to(device), device_ids=[rank]) if teacher_model else None
    requires_grad(model, True)  # DiT model requires gradients
    requires_grad(teacher_model, False)  # Teacher model does not require gradients

    hypernet = hypernet.to(device)
    requires_grad(hypernet, False)  # Hyperstructure does not require gradients

    # create diffusion and VAE
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    # Setup data:
    features_dir = f"{args.data_path}/imagenet256_features"
    labels_dir = f"{args.data_path}/imagenet256_labels"
    dataset = CustomDataset(features_dir, labels_dir, flip=0.5)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    print_rank_0(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader), eta_min=args.lr*0.5)
    opt_hyper = torch.optim.AdamW(hypernet.parameters(), lr=args.hyper_lr)

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights

    # Optionally resume
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=f'cuda:{device}')
        model.module.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        #sched.load_state_dict(checkpoint["sched"])
        print_rank_0(f"Resume training from {args.resume}")
        train_steps = checkpoint["train_steps"]
        start_epoch = checkpoint["epoch"]
        if "hypernet" in checkpoint and epoch < args.control_epochs:
            hypernet.load_state_dict(checkpoint["hypernet"])
            opt_hyper.load_state_dict(checkpoint["opt_hyper"])
            print_rank_0("Resumed hypernetwork state.")
        del checkpoint
    else:
        train_steps = 0
        start_epoch = 0

    if args.feature_kd:
        act_teacher = {}
        act_student = {}
        add_block_hooks_dit(teacher_model, act_teacher)
        add_block_hooks_dit(model, act_student)
    else:
        act_student, act_teacher = None, None

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_dit_loss = 0
    running_hypernet_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        sampler.set_epoch(epoch)
        # generate or freeze pruning mask
        if epoch < args.control_epochs:
            logger.info(f"[pruning_mask] is newly-generated, Hypernet together with DiT would be updated in epoch: {epoch}")
            hypernet.eval()
            with torch.no_grad():
                pruning_mask, router_out, _, _ = hypernet(torch.tensor(epoch, device=device))
                time_router = torch.argmax(router_out, dim=1).cpu().numpy()
            logger.info(f"Current time router: {time_router}")
        else:
            logger.info(f"[pruning_mask] is pre-fixed, only DiT weight would be updated in epoch: {epoch}")
            pruning_mask = pruning_mask  # freeze after control_epochs

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)
            
            # DiT training_step
            loss_dict = {}
            loss_dit, dit_loss_dict = dit_step(
                model, x, y, ema, opt, diffusion, pruning_mask, args, teacher_model=teacher_model,
                act_student=act_student, act_teacher=act_teacher
            )
            loss_dict.update(dit_loss_dict)
            running_dit_loss += loss_dit.item()

            # update hyperstructure
            if epoch < args.control_epochs:
                pruning_mask, loss_hypernet, hypernet_loss_dict = hypernet_step(
                    hypernet, model, x, y, opt_hyper, diffusion, args,
                    teacher_model=teacher_model, act_student=act_student, act_teacher=act_teacher
                )
                loss_dict.update(hypernet_loss_dict)
                loss_dict.update({"loss_hypernet": loss_hypernet.item()})
                running_hypernet_loss += loss_hypernet.item()

            # Log loss values:
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_dit_loss = torch.tensor(running_dit_loss / log_steps, device=device)
                dist.all_reduce(avg_dit_loss, op=dist.ReduceOp.SUM)
                avg_dit_loss = avg_dit_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_dit_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Log hypernet loss if applicable
                if epoch < args.control_epochs:
                    avg_hypernet_loss = torch.tensor(running_hypernet_loss / log_steps, device=device)
                    dist.all_reduce(avg_hypernet_loss, op=dist.ReduceOp.SUM)
                    avg_hypernet_loss = avg_hypernet_loss.item() / dist.get_world_size()
                    logger.info(f"(step={train_steps:07d}) Train Hypernet Loss: {avg_hypernet_loss:.4f}")
                    if rank == 0:
                        wandb.log({
                            "train_hypernet_loss": avg_hypernet_loss,
                            **{k: v for k, v in loss_dict.items() if k.startswith("loss_")}
                        })
                    running_hypernet_loss = 0
                # Reset monitoring variables:
                running_dit_loss = 0
                log_steps = 0
                start_time = time()
                if rank == 0:
                    wandb.log({
                        "train_loss": avg_dit_loss,
                        **{k: v for k, v in loss_dict.items() if k.startswith("loss_")}
                    })

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps,
                        "epoch": epoch
                    }
                    # Add hypernet state to the same checkpoint if in the search phase
                    if epoch < args.control_epochs:
                        checkpoint["hypernet"] = hypernet.state_dict()
                        checkpoint["opt_hyper"] = opt_hyper.state_dict()

                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                    # Also save the final pruning mask if the search phase is over
                    if epoch == args.control_epochs -1:
                        final_mask_path = f"{experiment_dir}/final_pruning_mask.pth"
                        torch.save(pruning_mask, final_mask_path)
                        logger.info(f"Saved final pruning mask to {final_mask_path}")

                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout

    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    # --- Core Model & Data ---
    parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save results and checkpoints.")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2", help="Which DiT model to use.")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256, help="Image resolution.")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes for conditioning.")
    parser.add_argument("--load-weight", type=str, default=None, help="Path to initial student model weights.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training from.")

    # --- Training & Optimization ---
    parser.add_argument("--epochs", type=int, default=1400, help="Total number of training epochs.")
    parser.add_argument("--global-batch-size", type=int, default=256, help="Total batch size across all GPUs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the DiT model.")
    parser.add_argument("--weight-decay", type=float, default=0, help="Weight decay for the optimizer.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers.")
    parser.add_argument("--global-seed", type=int, default=0, help="Global random seed.")

    # --- Distillation ---
    parser.add_argument("--distill", action='store_true', help="Enable knowledge distillation.")
    parser.add_argument("--distill_only", action='store_true', help="Use only distillation loss, no ground truth loss.")
    parser.add_argument("--distill_rate", type=float, default=1.0, help="Weight for the output distillation loss.")
    parser.add_argument("--feature_kd", action='store_true', help="Enable feature-based (block-level) distillation.")
    parser.add_argument("--feature_kd_rate", type=float, default=1.0, help="Weight for the feature distillation loss.")

    # --- Pruning & Hypernetwork ---

    parser.add_argument("--control_epochs", type=int, default=100, help="Number of epochs to train the hypernetwork.")
    parser.add_argument("--pruning_ratio", type=float, default=0.5, help="Target pruning ratio for the ratio loss.")
    parser.add_argument("--ratio_loss_rate", type=float, default=0.1, help="Weight for the pruning ratio loss.")
    parser.add_argument("--hyper_lr", type=float, default=1e-4, help="Learning rate for the hypernetwork.")
    parser.add_argument("--n_experts", type=int, default=10, help="Number of 'expert' masks for the hypernetwork to choose from.")
    parser.add_argument("--router_balance", action='store_true', help="Enable router balancing loss.")
    parser.add_argument("--router_balance_rate", type=float, default=0.01, help="Weight for the router balancing loss.")
    parser.add_argument("--T", type=int, default=1000, help="training timesteps.")
    
    # --- Logging & Checkpointing ---
    parser.add_argument("--prefix", type=str, default=None, help="A prefix for the experiment folder and wandb run name.")
    parser.add_argument("--log_every", type=int, default=100, help="Log training status every N steps.")
    parser.add_argument("--ckpt_every", type=int, default=50_000, help="Save a checkpoint every N steps.")

    args = parser.parse_args()
    main(args)
