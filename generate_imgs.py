import torch
import argparse
import safetensors
import os
import json
from tqdm.auto import tqdm
import numpy as np
import cv2

import pandas as pd
from diffusers import PNDMScheduler, StableDiffusionPipeline

from datasets import Dataset
from custom_unet.unet_2d_conditional import UNet2DConditionModelPruned
from custom_pipeline.sd_pipeline import StableDiffusionPrunedPipeline
from diffusers import AutoPipelineForText2Image


def process_cc3m(img_dir, caption_path, output_path, start, end):
    captions = pd.read_csv(caption_path, sep="\t", header=None, names=["caption", "link"], dtype={"caption": str, "link": str})

    images = os.listdir(img_dir)
    images = [os.path.join(img_dir, image) for image in images]

    sorted_pairs = sorted([(int(os.path.basename(image).split("_")[0]), image) for image in images])
    sorted_image_indices, sorted_images = zip(*sorted_pairs)
    sorted_image_indices = list(sorted_image_indices)
    sorted_images = list(sorted_images)

    captions = captions.iloc[sorted_image_indices].caption.values.tolist()

    """
    filtered_images = []
    filtered_captions = []
    for img, caption in zip(sorted_images[start:end], captions[start:end]):

        filename_npy = f"{img[:-4]}.npy"
        out_file = os.path.join(output_path, filename_npy)
        if os.path.exists(out_file):
            continue
        filtered_images.append(img)
        filtered_captions.append(caption)
    """

    dataset = Dataset.from_dict({"image": sorted_images[start:end], "text": captions[start:end]})
    return dataset


def process_coco_with_replica(img_dir, caption_path, output_path, start, end):
    caption_file = json.load(open(caption_path))
    images = []
    texts = []
    image_ids = {}
    for temp in caption_file['annotations'][start:end]:
        # process the several captions for one image
        cur_image_id = temp['image_id']
        if cur_image_id in image_ids:
            image_ids[cur_image_id] += 1
        else:
            image_ids[cur_image_id] = 0

        anno_index = image_ids[cur_image_id]
        # Build the final .npy filename you will save to:
        filename_npy = f"{cur_image_id:012d}_{anno_index}.npy"
        out_file = os.path.join(output_path, filename_npy)

        # If it already exists in output_path, we skip it
        if os.path.exists(out_file):
            # Already generated => skip
            continue

        img_path = os.path.join(img_dir, "%012d_%d.jpg" % (cur_image_id, image_ids[cur_image_id]))
        caption = temp['caption']
        images.append(img_path)
        texts.append(caption)
    dataset = Dataset.from_dict({"image": images, "text": texts})
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="test script.")
    parser.add_argument('--img_dir', type=str, default="/data/mscoco/val2017")
    parser.add_argument('--caption_path', type=str, default="/data/mscoco/annotations/captions_val2017.json")
    parser.add_argument('--pretrained_model_name_or_path', type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--pruning_mask_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=16, help="Batch size (per device) for the testing dataloader."
    )
    parser.add_argument("--dataset", type=str, default="coco", help="The dataset to use for testing. One of 'cc3m', 'coco'.")
    parser.add_argument("--seed", type=int, default=43, help="A seed for reproducible testing.")
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument('--time_interval', type=int, default=None, help="number of time intervals to do the dynamic pruning")
    # parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--tome", action='store_true', help='whether to use token merging')
    parser.add_argument("--tome_ratio", type=float, default=0.5)

    args = parser.parse_args()
    return args


def main(args):
    os.makedirs(args.output_path, exist_ok=True)

    # If passed along, set the seed now.
    if args.seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
    else:
        generator = None

    if args.checkpoint_path is not None:
        print("=====> Intialization pre-trained: '{}' from Huggingface <=====".format(args.pretrained_model_name_or_path))
        unet = UNet2DConditionModelPruned.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            down_block_types=['CrossAttnDownBlock2DPruned', 'CrossAttnDownBlock2DPruned', 'CrossAttnDownBlock2DPruned', 'DownBlock2DPruned'],
            mid_block_type='UNetMidBlock2DCrossAttnPruned',
            up_block_types=['UpBlock2DPruned', 'CrossAttnUpBlock2DPruned', 'CrossAttnUpBlock2DPruned', 'CrossAttnUpBlock2DPruned'],
        )

        print("=====> Load checkpoint: '{}' <=====".format(args.checkpoint_path))
        state_dict = safetensors.torch.load_file(os.path.join(args.checkpoint_path, "unet", "diffusion_pytorch_model.safetensors"))

        unet.load_state_dict(state_dict)

        print("=====> Load pruning_mask: '{}' <=====".format(args.pruning_mask_path))
        pruning_mask = torch.load(args.pruning_mask_path, map_location="cuda")

    print("=====> Load pipeline. <=====")
    if args.pretrained_model_name_or_path == "stabilityai/sd-turbo":
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo")
    else:
        noise_scheduler = PNDMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
        if args.checkpoint_path is not None:
            pipe = StableDiffusionPrunedPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unet,
                scheduler=noise_scheduler,
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                scheduler=noise_scheduler,
            )
    if args.tome:
        tomesd.apply_patch(pipe, ratio=args.tome_ratio)
    pipe.to("cuda")

    if args.time_interval is not None:
        noise_scheduler = pipe.scheduler
        interval_size = noise_scheduler.config.num_train_timesteps // args.time_interval

    print("=====> Load {} dataset <=====".format(args.dataset))
    if args.dataset == "cc3m":
        dataset = process_cc3m(args.img_dir, args.caption_path, args.output_path, args.start, args.end)
    elif args.dataset == "coco":
        dataset = process_coco_with_replica(args.img_dir, args.caption_path, args.output_path, args.start, args.end)
    elif args.dataset == "imagereward":
        with open(args.caption_path, "r") as f:
            entries = json.load(f)
            print(f"Loaded {len(entries)} prompts from {args.caption_path}")
            
        # ─── Generate 10 images per prompt ─────────────────────────────────────────
        for entry in tqdm(entries, desc="Prompts"):
            pid    = entry["id"]
            prompt = entry["prompt"]
    
            for i in range(10):
                seed = args.seed + i
                generator = torch.Generator(device="cuda").manual_seed(seed)
                if args.checkpoint_path is not None:
                    image = pipe(
                        prompt,
                        num_inference_steps=args.num_inference_steps,
                        generator=generator,
                        pruning_mask=pruning_mask,
                    ).images[0]
                else:
                    image = pipe(
                        prompt,
                        num_inference_steps=args.num_inference_steps,
                        generator=generator,
                    ).images[0]                    
                # Save directly under output_path, named with id and index
                filename = f"{pid}_{i}.png"
                image.save(os.path.join(args.output_path, filename))
        
        return

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # DataLoaders creation:
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    for batch in tqdm(test_dataloader):
        if args.checkpoint_path is not None:
            images = pipe(batch[args.caption_column],
                          num_inference_steps=args.num_inference_steps,
                          generator=generator,
                          pruning_mask=pruning_mask,
                          interval_size=interval_size if args.time_interval is not None else None,
                          output_type="np",
                          height=args.height,
                          width=args.width
                          ).images
        elif args.pretrained_model_name_or_path == "stabilityai/sd-turbo":
            images = pipe(batch[args.caption_column],
                          num_inference_steps=args.num_inference_steps,
                          generator=generator,
                          guidance_scale=0.0,
                          output_type="np",
                          height=args.height,
                          width=args.width
                          ).images
        else:
            images = pipe(batch[args.caption_column],
                          num_inference_steps=args.num_inference_steps,
                          generator=generator,
                          output_type="np",
                          height=args.height,
                          width=args.width
                          ).images

        for i, img in enumerate(batch[args.image_column]):
            img_name = img.split("/")[-1]
            if img_name.endswith(".jpg") or img_name.endswith(".npy") or img_name.endswith(".png"):
                img_name = img_name[:-4]
            img_path = os.path.join(args.output_path, f"{img_name}.npy")
            gen_img = images[i]
            gen_img = gen_img * 255
            gen_img = gen_img.astype(np.uint8)
            gen_img = cv2.resize(gen_img, (256, 256))
            np.save(img_path, gen_img)
            
    return


if __name__ == "__main__":
    args = parse_args()
    main(args)

