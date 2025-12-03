import torch
import torch.nn.functional as F

from diffusers.training_utils import compute_dream_and_update_latents, compute_snr
from utils.prune_utils import select_pruning_mask
from utils.loss_utils import match_loss, router_balance_loss


def generate_timestep_weights(args, num_timesteps):
    weights = torch.ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)

    if args.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
            )
        if range_end > num_timesteps:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
            )
        bias_indices = slice(range_begin, range_end)
    else:  # 'none' or any other string
        return weights
    if args.timestep_bias_multiplier <= 0:
        return ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] *= args.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return weights

#-----------------------------------------------------------------#

#-----------------------------------------------------------------#

def preprocess_input(vae, text_encoder, batch, noise_scheduler, weight_dtype, args, interval_size=None):
    # Convert images to latent space
    latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    if args.noise_offset:
        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
        noise += args.noise_offset * torch.randn(
            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
        )
    if args.input_perturbation:
        new_noise = noise + args.input_perturbation * torch.randn_like(noise)
    bsz = latents.shape[0]
    # Sample a random timestep for each image
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    # interval_idx = timesteps // interval_size

    # First sample the interval_idx and make sure the timesteps in the same interval
    # interval_idx = torch.randint(0, noise_scheduler.config.num_train_timesteps // interval_size, (1,), device=latents.device) 
    # interval_start = interval_idx * interval_size
    # interval_end = min((interval_idx + 1) * interval_size, noise_scheduler.config.num_train_timesteps)
    # timesteps = torch.randint(interval_start.item(), interval_end.item(), (bsz,), device=latents.device).long()
    # interval_idx = interval_idx.repeat(bsz)

    timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    if args.input_perturbation:
        noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
    else:
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Get the text embedding for conditioning
    encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

    # Get the target for loss depending on the prediction type
    if args.prediction_type is not None:
        # set prediction_type of scheduler if defined
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    
    return noise, noisy_latents, timesteps, encoder_hidden_states, target
    
#-----------------------------------------------------------------#
#-----------------------------------------------------------------#

def sd_step(unet, vae, text_encoder, batch, pruning_mask, optimizer, noise_scheduler, accelerator,
            lr_scheduler, weight_dtype, args, interval_size=None, teacher_unet=None, lora_layers=None,
            act_student=None, act_teacher=None):
    # Unfreeze unet parameters
    unet.train()
    if args.lora:
        for param in lora_layers:
            param.requires_grad = True
    else:
        unet.requires_grad_(True)

    loss_dict = {}
    with accelerator.accumulate(unet):

        noise, noisy_latents, timesteps, encoder_hidden_states, target = preprocess_input(
            vae, text_encoder, batch, noise_scheduler, weight_dtype, args
        )

        if args.dream_training:
            noisy_latents, target = compute_dream_and_update_latents(
                unet,
                noise_scheduler,
                timesteps,
                noise,
                noisy_latents,
                target,
                encoder_hidden_states,
                args.dream_detail_preservation,
            )

        # Predict the noise residual and compute loss
        selected_mask = select_pruning_mask(pruning_mask, timesteps)
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False, 
                          pruning_mask=selected_mask)[0]
        
        if args.distill:
            with torch.no_grad():
                full_model_pred = teacher_unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0].detach()
            distill_loss = F.mse_loss(model_pred.float(), full_model_pred.float(), reduction="mean")
            loss_dict['loss_unet_distill'] = distill_loss.item()

            feature_loss = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
            if args.feature_kd:
                for key in act_student.keys():
                    feature_loss += F.mse_loss(act_student[key].float(),
                                               act_teacher[key].detach().float(), reduction="mean")
        
                feature_loss = feature_loss / len(act_student.keys())
                loss_dict['loss_unet_block'] = feature_loss.item()

        if not args.distill_only:
            if args.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                    dim=1
                )[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

                loss_dict['loss_unet_sd'] = loss.item()
            if args.distill:
                total_loss = loss + args.distill_rate * distill_loss + args.feature_kd_rate * feature_loss
            else:
                total_loss = loss
        else:
            total_loss = distill_loss + args.feature_kd_rate * feature_loss

        # Gather the losses across all processes for logging (if we use distributed training).
        avg_loss = accelerator.gather(total_loss.repeat(args.train_batch_size)).mean()
        loss_dict['loss_unet'] = avg_loss.item()

        # Backpropagate
        accelerator.backward(total_loss)
        if accelerator.sync_gradients:
            if args.lora:
                params_to_clip = lora_layers
            else:
                params_to_clip = unet.parameters()
            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    
    return total_loss, avg_loss, loss_dict
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#

def hypernet_step(hypernet, unet, vae, text_encoder, batch, optimizer, noise_scheduler, accelerator,
                  lr_scheduler, weight_dtype, pruning_contribution, pruning_ratio, args, interval_size=None,
                  teacher_unet=None, lora_layers=None, act_student=None, act_teacher=None):
    # a) freeze unet & unfreeze hypernet()
    unet.eval()
    if args.lora:
        for param in lora_layers:
            param.requires_grad = False
    else:
        unet.requires_grad_(False)
    hypernet.train()
    hypernet.requires_grad_(True)
    hypernet_unwrapped = hypernet.module if hasattr(hypernet, "module") else hypernet

    loss_dict = {}
    with accelerator.accumulate(hypernet):
        # b) hypernet.forward() (get logits instead of binary mask for hypernet() training)
        # Generate pruning_mask using hypernet
        dummy_input = torch.tensor(0).to(accelerator.device)
        mask_vec, router_out, router_logits, expert_feat = hypernet(dummy_input)
        # mask_vec = hypernet()
        pruning_mask = hypernet_unwrapped.transform_output(mask_vec)

        # c) subnet of sd forward() with current pruning mask
        noise, noisy_latents, timesteps, encoder_hidden_states, target = preprocess_input(
            vae, text_encoder, batch, noise_scheduler, weight_dtype, args
        )

        if args.dream_training:
            noisy_latents, target = compute_dream_and_update_latents(
                unet,
                noise_scheduler,
                timesteps,
                noise,
                noisy_latents,
                target,
                encoder_hidden_states,
                args.dream_detail_preservation,
            )

        # Predict the noise residual and compute loss
        selected_mask = select_pruning_mask(pruning_mask, timesteps)
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False, 
                          pruning_mask=selected_mask)[0]

        if args.distill:
            with torch.no_grad():
                full_model_pred = teacher_unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0].detach()
            distill_loss = F.mse_loss(model_pred.float(), full_model_pred.float(), reduction="mean")

            feature_loss = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
            if args.feature_kd:
                for key in act_student.keys():
                    feature_loss += F.mse_loss(act_student[key].float(),
                                               act_teacher[key].detach().float(), reduction="mean")
        
                feature_loss = feature_loss / len(act_student.keys())
        
        if not args.distill_only:
            if args.snr_gamma is None:
                unet_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                    dim=1
                )[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)

                unet_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                unet_loss = unet_loss.mean(dim=list(range(1, len(unet_loss.shape)))) * mse_loss_weights
                unet_loss = unet_loss.mean()

            if args.distill:
                unet_loss = unet_loss + args.distill_rate * distill_loss + args.feature_kd_rate * feature_loss
        else:
            unet_loss = distill_loss + args.feature_kd_rate * feature_loss

        # d) mask constrain: total pruning ratio
        mask_ratio = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
        for ratio_list, mask_list in zip(pruning_contribution, pruning_mask):
            for r, mask in zip(ratio_list, mask_list):
            # 'mask' has shape (T, original_shape)
            # Sum all elements of the weighted tensor and add to total
                mask_ratio += (mask.mean(dim=0) * r).sum()

        # i) the total mask ratio is close to 0.5
        ratio_loss  = match_loss(mask_ratio, pruning_ratio)
        loss_dict['loss_ratio'] = ratio_loss.item()

        hyper_loss = unet_loss + args.ratio_loss_rate * ratio_loss

        if args.router_balance:
            router_loss = router_balance_loss(router_logits, router_out, args.n_expert)
            loss_dict['loss_router_balance'] = router_loss.item()
            hyper_loss = hyper_loss + args.router_balance_rate * router_loss

        # e) sum the loss
        # Gather the losses across all processes for logging (if we use distributed training).
        avg_loss = accelerator.gather(hyper_loss.repeat(args.train_batch_size)).mean()

        # Backpropagate
        accelerator.backward(hyper_loss)
        if accelerator.sync_gradients:
            params_to_clip = hypernet.parameters()
            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    hypernet.eval()
    with torch.no_grad():
        # mask_vec = hypernet()  
        mask_vec, router_out, _, _ = hypernet(dummy_input)  
        pruning_mask = hypernet_unwrapped.transform_output(mask_vec)

    return pruning_mask, mask_vec, hyper_loss, avg_loss, loss_dict
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#

def preprocess_input_sdxl(batch, noise_scheduler, accelerator, weight_dtype, args, interval_size=None):
    # Sample noise that we'll add to the latents
    model_input = batch["model_input"]
    if model_input.device != accelerator.device:
        model_input = model_input.to(accelerator.device)
    noise = torch.randn_like(model_input)
    if args.noise_offset:
        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
        noise += args.noise_offset * torch.randn(
            (model_input.shape[0], model_input.shape[1], 1, 1), device=model_input.device
        )

    # Determine batch size
    bsz = model_input.shape[0]
    if args.timestep_bias_strategy == "none":
        # Sample a random timestep for each image without bias.
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device).long()

        # interval_idx = torch.randint(0, noise_scheduler.config.num_train_timesteps // interval_size, (1,), device=model_input.device) 
        # interval_start = interval_idx * interval_size
        # interval_end = min((interval_idx + 1) * interval_size, noise_scheduler.config.num_train_timesteps)
        # timesteps = torch.randint(interval_start.item(), interval_end.item(), (bsz,), device=model_input.device).long()
        # interval_idx = interval_idx.repeat(bsz)
    else:
        # Sample a random timestep for each image, potentially biased by the timestep weights.
        # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
        weights = generate_timestep_weights(args, noise_scheduler.config.num_train_timesteps).to(
            model_input.device
        )
        timesteps = torch.multinomial(weights, bsz, replacement=True).long()

    # Add noise to the model input according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps).to(dtype=weight_dtype)

    # time ids
    def compute_time_ids(original_size, crops_coords_top_left):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (args.resolution, args.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], device=accelerator.device, dtype=weight_dtype)
        return add_time_ids

    add_time_ids = torch.cat(
        [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
    )

    # Predict the noise residual
    unet_added_conditions = {"time_ids": add_time_ids}
    prompt_embeds = batch["prompt_embeds"]
    if prompt_embeds.device != accelerator.device:
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)

    pooled_prompt_embeds = batch["pooled_prompt_embeds"]
    if pooled_prompt_embeds.device != accelerator.device:
        pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)

    unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})

    # loss for subnet sd
    # Get the target for loss depending on the prediction type
    if args.prediction_type is not None:
        # set prediction_type of scheduler if defined
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(model_input, noise, timesteps)
    elif noise_scheduler.config.prediction_type == "sample":
        # We set the target to latents here, but the model_pred will return the noise sample prediction.
        target = model_input
        # # We will have to subtract the noise residual from the prediction to get the target sample.
        # model_pred = model_pred - noise
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    return noise, noisy_model_input, timesteps, prompt_embeds, unet_added_conditions, target

#-----------------------------------------------------------------#

#-----------------------------------------------------------------#

def sdxl_step(unet, batch, pruning_mask, optimizer, noise_scheduler, accelerator, lr_scheduler, weight_dtype, args,
              interval_size=None, teacher_unet=None, lora_layers=None, act_student=None, act_teacher=None):
    # Unfreeze unet parameters
    unet.train()
    if args.lora:
        for param in lora_layers:
            param.requires_grad = True
    else:
        unet.requires_grad_(True)

    loss_dict = {}
    with accelerator.accumulate(unet):

        # Get the inputs to the unet
        noise, noisy_model_input, timesteps, prompt_embeds, unet_added_conditions, target = preprocess_input_sdxl(
            batch, noise_scheduler, accelerator, weight_dtype, args
        )

        # Prediction with current pruning masks
        selected_mask = select_pruning_mask(pruning_mask, timesteps)
        model_pred = unet(noisy_model_input, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions,
                            return_dict=False, pruning_mask=selected_mask)[0]

        if noise_scheduler.config.prediction_type == "sample":
            model_pred = model_pred - noise

        if args.distill:
            with torch.no_grad():
                full_model_pred = teacher_unet(noisy_model_input, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions,
                                                return_dict=False)[0]
                if noise_scheduler.config.prediction_type == "sample":
                    full_model_pred = full_model_pred - noise
            distill_loss = F.mse_loss(model_pred.float(), full_model_pred.float(), reduction="mean")
            loss_dict['loss_unet_distill'] = distill_loss.item()

            feature_loss = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
            if args.feature_kd:
                for key in act_student.keys():
                    feature_loss += F.mse_loss(act_student[key].float(),
                                                act_teacher[key].detach().float(), reduction="mean")
        
                feature_loss = feature_loss / len(act_student.keys())
                loss_dict['loss_unet_block'] = feature_loss.item()

        if not args.distill_only:
            if args.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

                loss_dict['loss_unet_sd'] = loss.item()
            if args.distill:
                total_loss = loss + args.distill_rate * distill_loss + args.feature_kd_rate * feature_loss
            else:
                total_loss = loss
        else:
            total_loss = distill_loss + args.feature_kd_rate * feature_loss

        # Gather the losses across all processes for logging (if we use distributed training).
        avg_loss = accelerator.gather(total_loss.repeat(args.train_batch_size)).mean()
        loss_dict['loss_unet'] = avg_loss.item()

        # Backpropagate
        accelerator.backward(total_loss)
        if accelerator.sync_gradients:
            if args.lora:
                params_to_clip = lora_layers
            else:
                params_to_clip = unet.parameters()
            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return total_loss, avg_loss, loss_dict

#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
# step_wise forward() for hypernet() param_tuning
def hypernet_sdxl_step(hypernet, unet, batch, optimizer, noise_scheduler, accelerator, lr_scheduler, weight_dtype, 
                       pruning_contribution, pruning_ratio, args, interval_size=None, teacher_unet=None,
                       lora_layers=None, act_student=None, act_teacher=None):
    # a) freeze unet & unfreeze hypernet()
    unet.eval()
    if args.lora:
        for param in lora_layers:
            param.requires_grad = False
    else:
        unet.requires_grad_(False)
    hypernet.train()
    hypernet.requires_grad_(True)
    hypernet_unwrapped = hypernet.module if hasattr(hypernet, "module") else hypernet

    loss_dict = {}
    with accelerator.accumulate(hypernet):
        # b) hypernet.forward() (get logits instead of binary mask for hypernet() training)
        # Generate pruning_mask using hypernet
        dummy_input = torch.tensor(0).to(accelerator.device)
        mask_vec, router_out, router_logits, expert_feat = hypernet(dummy_input)
        # mask_vec = hypernet()
        pruning_mask = hypernet_unwrapped.transform_output(mask_vec)

        # c) subnet of sd forward() with current pruning mask
        noise, noisy_model_input, timesteps, prompt_embeds, unet_added_conditions, target = preprocess_input_sdxl(
            batch, noise_scheduler, accelerator, weight_dtype, args
        )

        selected_mask = select_pruning_mask(pruning_mask, timesteps)
        model_pred = unet(noisy_model_input, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions, 
                            return_dict=False, pruning_mask=selected_mask)[0]
        
        if noise_scheduler.config.prediction_type == "sample":
            model_pred = model_pred - noise

        if args.distill:
            with torch.no_grad():
                full_model_pred = teacher_unet(noisy_model_input, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions,
                                                return_dict=False)[0]
                if noise_scheduler.config.prediction_type == "sample":
                    full_model_pred = full_model_pred - noise
            distill_loss = F.mse_loss(model_pred.float(), full_model_pred.float(), reduction="mean")

            feature_loss = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
            if args.feature_kd:
                for key in act_student.keys():
                    feature_loss += F.mse_loss(act_student[key].float(),
                                                act_teacher[key].detach().float(), reduction="mean")
        
                feature_loss = feature_loss / len(act_student.keys())
            
        if not args.distill_only:
            if args.snr_gamma is None:
                unet_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)

                unet_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                unet_loss = unet_loss.mean(dim=list(range(1, len(unet_loss.shape)))) * mse_loss_weights
                unet_loss = unet_loss.mean()

            if args.distill:
                unet_loss = unet_loss + args.distill_rate * distill_loss + args.feature_kd_rate * feature_loss
        else:
            unet_loss = distill_loss + args.feature_kd_rate * feature_loss

        # d) mask constrain: total pruning ratio
        mask_ratio = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
        for ratio_list, mask_list in zip(pruning_contribution, pruning_mask):
            for r, mask in zip(ratio_list, mask_list):
            # Sum all elements of the weighted tensor and add to total
                mask_ratio += (mask.mean(dim=0) * r).sum()

        # i) the total mask ratio is close to 0.5
        ratio_loss  = match_loss(mask_ratio, pruning_ratio)
        loss_dict['loss_ratio'] = ratio_loss.item()

        hyper_loss = unet_loss + args.ratio_loss_rate * ratio_loss

        if args.router_balance:
            router_loss = router_balance_loss(router_logits, router_out, args.n_expert)
            loss_dict['loss_router_balance'] = router_loss.item()
            hyper_loss = hyper_loss + args.router_balance_rate * router_loss

        # e) sum the loss
        # Gather the losses across all processes for logging (if we use distributed training).
        avg_loss = accelerator.gather(hyper_loss.repeat(args.train_batch_size)).mean()

        # Backpropagate
        accelerator.backward(hyper_loss)
        if accelerator.sync_gradients:
            params_to_clip = hypernet.parameters()
            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    hypernet.eval()
    with torch.no_grad():
        # mask_vec = hypernet()  
        mask_vec, router_out, _, _ = hypernet(dummy_input)  
        pruning_mask = hypernet_unwrapped.transform_output(mask_vec)

    return pruning_mask, mask_vec, hyper_loss, avg_loss, loss_dict

