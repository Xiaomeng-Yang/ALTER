import os
import json
import random
import torch
import functools
import gc
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from datasets import concatenate_datasets, load_dataset, Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from diffusers.utils.import_utils import is_torch_npu_available


if is_torch_npu_available():
    import torch_npu

    torch.npu.config.allow_internal_format = False


def fetch_and_convert_images(url_list):
    images = []
    for url in url_list:
        try:
            # Fetch image from URL
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad status codes
            
            # Open image as RGB
            image = Image.open(BytesIO(response.content)).convert("RGB")
            images.append(image)
        except Exception as e:
            # print(f"Error processing URL {url}: {e}")
            continue
            # images.append(None)  # Append None if the image cannot be processed
    return images


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(batch, text_encoders, tokenizers, proportion_empty_prompts, caption_column, is_train=True):
    prompt_embeds_list = []
    if caption_column == 'json':
        prompt_batch = [json['caption'] for json in batch[caption_column]]
    else:
        prompt_batch = batch[caption_column]

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
                return_dict=False,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds.cpu(), "pooled_prompt_embeds": pooled_prompt_embeds.cpu()}


def compute_vae_encodings(batch, vae):
    images = batch.pop("pixel_values")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor

    # There might have slightly performance improvement
    # by changing model_input.cpu() to accelerator.gather(model_input)
    return {"model_input": model_input.cpu()}


class DataProcessor_SDXL:
    def __init__(self, args):
        self.args = args
        self.train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_crop = transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    
    def _preprocess_train(self, examples, image_column):
        if isinstance(examples[image_column][0], str):
            images = fetch_and_convert_images(examples[image_column])
        else:
            images = [image.convert("RGB") for image in examples[image_column]]
        # image aug
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = self.train_resize(image)
            if self.args.random_flip and random.random() < 0.5:
                # flip
                image = self.train_flip(image)
            if self.args.center_crop:
                y1 = max(0, int(round((image.height - self.args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - self.args.resolution) / 2.0)))
                image = self.train_crop(image)
            else:
                y1, x1, h, w = self.train_crop.get_params(image, (self.args.resolution, self.args.resolution))
                image = crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = self.train_transforms(image)
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        return examples
    
    def _collate_fn(self, examples):
        model_input = torch.stack([torch.tensor(example["model_input"]) for example in examples])
        original_sizes = [example["original_sizes"] for example in examples]
        crop_top_lefts = [example["crop_top_lefts"] for example in examples]
        prompt_embeds = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])
        pooled_prompt_embeds = torch.stack([torch.tensor(example["pooled_prompt_embeds"]) for example in examples])

        return {
            "model_input": model_input,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
        }
    
    def get_valid_dataset(self, dataset_name, dataset_config_name=None, cache_dir=None):
        dataset = load_dataset(dataset_name,
                               dataset_config_name,
                               cache_dir)
        train_dataset = dataset["train"]
        valid_indices = []
        for idx in range(len(train_dataset)):
            try:
                example = train_dataset[idx]
                valid_indices.append(idx)
            except:
                continue

        filtered_dataset = train_dataset.select(valid_indices)
        return filtered_dataset
    
    def create_dataloader(self, accelerator, text_encoder_one, text_encoder_two, 
                          tokenizer_one, tokenizer_two, vae, vae_path):
        # Let's first compute all the embeddings so that we can free up the text encoders
        # from memory. We will pre-compute the VAE encodings too.
        text_encoders = [text_encoder_one, text_encoder_two]
        tokenizers = [tokenizer_one, tokenizer_two]
        compute_embeddings_fn = functools.partial(
            encode_prompt,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            proportion_empty_prompts=self.args.proportion_empty_prompts,
            caption_column=self.args.caption_column,
        )
        compute_vae_encodings_fn = functools.partial(compute_vae_encodings, vae=vae)

        image_column = self.args.image_column 
        caption_column = self.args.caption_column

        def preprocess_fn(examples):
            return self._preprocess_train(examples, image_column)
        
        dataset = self.get_valid_dataset(
            self.args.dataset_name,
            cache_dir=self.args.cache_dir,
        )
        column_names = dataset.column_names

        with accelerator.main_process_first():
            if self.args.max_train_samples is not None:
                dataset = dataset.shuffle(seed=self.args.seed).select(range(self.args.max_train_samples))
            # Set the training transforms
            train_dataset = dataset.with_transform(preprocess_fn)

            from datasets.fingerprint import Hasher
            # fingerprint used by the cache for the other processes to load the result
            # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401

            # Build a dictionary that includes all relevant
            fingerprint_input = {
                "pretrained_model_name_or_path": self.args.pretrained_model_name_or_path,
                "pretrained_vae_model_name_or_path": self.args.pretrained_vae_model_name_or_path,
                "revision": self.args.revision,
                "variant": self.args.variant,
                "dataset_name": self.args.dataset_name,
                "image_column": self.args.image_column,
                "caption_column": self.args.caption_column,
                "max_train_samples": self.args.max_train_samples,
                "proportion_empty_prompts": self.args.proportion_empty_prompts,
                "cache_dir": self.args.cache_dir,
                "seed": self.args.seed,
                "resolution": self.args.resolution,
                "center_crop": self.args.center_crop,
                "random_flip": self.args.random_flip,
                "train_batch_size": self.args.train_batch_size,
                "dataloader_num_workers": self.args.dataloader_num_workers,
                "mixed_precision": self.args.mixed_precision,
            }

            new_fingerprint = Hasher.hash(fingerprint_input)
            new_fingerprint_for_vae = Hasher.hash((vae_path, fingerprint_input))
            train_dataset_with_embeddings = train_dataset.map(
                compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint
            )
            train_dataset_with_vae = train_dataset.map(
                compute_vae_encodings_fn,
                batched=True,
                batch_size=self.args.train_batch_size,
                new_fingerprint=new_fingerprint_for_vae,
            )
            precomputed_dataset = concatenate_datasets(
                [train_dataset_with_embeddings, train_dataset_with_vae.remove_columns(column_names)], axis=1
            )
            precomputed_dataset = precomputed_dataset.with_transform(preprocess_fn)

        del compute_vae_encodings_fn, compute_embeddings_fn, text_encoder_one, text_encoder_two
        del text_encoders, tokenizers, vae
        gc.collect()
        if is_torch_npu_available():
            torch_npu.npu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        num_examples = len(precomputed_dataset)
        train_dataloader = torch.utils.data.DataLoader(
            precomputed_dataset,
            shuffle=True,
            collate_fn=self._collate_fn,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        return num_examples, train_dataloader


class DataProcessor:
    def __init__(self, args):
        self.args = args
        # Preprocessing the datasets.
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    def _tokenize_captions(self, examples, caption_column, tokenizer, is_train=True):
        if caption_column == 'json':
            prompt_batch = [json['caption'] for json in examples[caption_column]]
        else:
            prompt_batch = examples[caption_column]

        captions = []
        for caption in prompt_batch:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    def _preprocess_train(self, tokenizer, examples, image_column, caption_column):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [self.train_transforms(image) for image in images]
        examples["input_ids"] = self._tokenize_captions(examples, caption_column, tokenizer)
        return examples
    
    def _collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    
    def get_valid_dataset(self, dataset_name, dataset_config_name=None, cache_dir=None):
        dataset = load_dataset(dataset_name,
                               dataset_config_name,
                               cache_dir)
        train_dataset = dataset["train"]
        valid_indices = []
        for idx in range(len(train_dataset)):
            try:
                example = train_dataset[idx]
                valid_indices.append(idx)
            except:
                continue

        filtered_dataset = train_dataset.select(valid_indices)
        return filtered_dataset
    
    def create_dataloader(self, tokenizer, accelerator):
        image_column = self.args.image_column 
        caption_column = self.args.caption_column

        def preprocess_fn(examples):
            return self._preprocess_train(tokenizer, examples, image_column, caption_column)
        
        dataset = self.get_valid_dataset(
            self.args.dataset_name,
            cache_dir=self.args.cache_dir,
        )

        with accelerator.main_process_first():
            if self.args.max_train_samples is not None:
                dataset = dataset.shuffle(seed=self.args.seed).select(range(self.args.max_train_samples))
            # Set the training transforms
            train_dataset = dataset.with_transform(preprocess_fn)

        num_examples = len(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=self._collate_fn,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        return num_examples, train_dataloader


def preprocess_coco(img_dir, caption_path):
    caption_file = json.load(open(caption_path))
    images = []
    texts = []
    for temp in caption_file['annotations']:
        img_path = os.path.join(img_dir, "%012d.jpg" % temp['image_id'])
        caption = temp['caption']
        images.append(img_path)
        texts.append(caption)
    dataset = Dataset.from_dict({"image": images, "text": texts})
    return dataset


class DataProcessor_SD3:
    def __init__(self, args):
        self.args = args
        # Preprocessing the datasets.
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    def _tokenize_captions(self, examples, caption_column, tokenizer, is_train=True):
        if caption_column == 'json':
            prompt_batch = [json['caption'] for json in examples[caption_column]]
        else:
            prompt_batch = examples[caption_column]

        captions = []
        for caption in prompt_batch:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    def _preprocess_train(self, tokenizer, tokenizer_2, tokenizer_3, examples, image_column, caption_column):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [self.train_transforms(image) for image in images]
        examples["input_ids"] = self._tokenize_captions(examples, caption_column, tokenizer)
        examples["input_ids_2"] = self._tokenize_captions(examples, caption_column, tokenizer_2)
        examples["input_ids_3"] = self._tokenize_captions(examples, caption_column, tokenizer_3)
        return examples
    
    def _collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        input_ids_2 = torch.stack([example["input_ids_2"] for example in examples])
        input_ids_3 = torch.stack([example["input_ids_3"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids, "input_ids_2": input_ids_2, "input_ids_3": input_ids_3}
    
    def get_valid_dataset(self, dataset_name, dataset_config_name=None, cache_dir=None):
        dataset = load_dataset(dataset_name,
                               dataset_config_name,
                               cache_dir)
        train_dataset = dataset["train"]
        valid_indices = []
        for idx in range(len(train_dataset)):
            try:
                example = train_dataset[idx]
                valid_indices.append(idx)
            except:
                continue

        filtered_dataset = train_dataset.select(valid_indices)
        return filtered_dataset
    
    def create_dataloader(self, tokenizer, tokenizer_2, tokenizer_3, accelerator):
        image_column = self.args.image_column 
        caption_column = self.args.caption_column

        def preprocess_fn(examples):
            return self._preprocess_train(tokenizer, tokenizer_2, tokenizer_3, examples, image_column, caption_column)
        
        dataset = self.get_valid_dataset(
            self.args.dataset_name,
            cache_dir=self.args.cache_dir,
        )

        with accelerator.main_process_first():
            if self.args.max_train_samples is not None:
                dataset = dataset.shuffle(seed=self.args.seed).select(range(self.args.max_train_samples))
            # Set the training transforms
            train_dataset = dataset.with_transform(preprocess_fn)

        num_examples = len(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=self._collate_fn,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        return num_examples, train_dataloader

