import argparse
import os
import logging
import json
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import torch
# import clip
import open_clip
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def get_index(filename):
    # Use regular expressions to find all digits in the filename
    # Assumes filenames are like '000000.jpg', 'image_000001.png', etc.
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        return -1  # Return -1 or any default value if no digits are found
    

class ImageCaptionDataset(Dataset):
    def __init__(self, annotations, generation_path, preprocess):
        self.image_entries = []
        image_counts = {}

        for temp in annotations:
            cur_image_id = temp['image_id']
            image_counts[cur_image_id] = image_counts.get(cur_image_id, -1) + 1
            img_path = os.path.join(
                generation_path, 
                f"{cur_image_id:012d}_{image_counts[cur_image_id]}.npy"
            )
            caption = temp['caption']
            self.image_entries.append((img_path, caption))
        
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_entries)

    def __getitem__(self, idx):
        img_path, caption = self.image_entries[idx]
        data = np.load(img_path)
        with Image.fromarray(data).convert('RGB') as img:
            image_tensor = self.preprocess(img)
        return image_tensor, caption


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--generation_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, choices=["coco", "partiprompt"], 
                        help="Specify the dataset name: 'coco' or 'partiprompt'.")
    parser.add_argument('--prompt_path', type=str, default="/work/yanzhi_group/yang.xiaome/dataset/T2I/mscoco/annotations/captions_val2017.json")
    parser.add_argument('--clip_model', type=str, default="hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    return args


def get_clip_score(args, device):
    print(f"Loading CLIP model: {args.clip_model}")

    # model, preprocess = clip.load(args.clip_model, device=device)
    model, preprocess = open_clip.create_model_from_pretrained(args.clip_model, device=device)
    tokenizer = open_clip.get_tokenizer(args.clip_model)

    # Assuming args is defined and has prompt_path, generation_path, batch_size, etc.
    caption_file = json.load(open(args.prompt_path))
    annotations = caption_file['annotations']

    # Create the dataset
    dataset = ImageCaptionDataset(annotations, args.generation_path, preprocess)

    # Create the DataLoader
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    clip_scores = []
    logit_scale = model.logit_scale.exp()

    with torch.no_grad():
        for image, caption in tqdm(test_dataloader):
            img_features = model.encode_image(image.to(device))
            text_features = model.encode_text(tokenizer(caption).to(device))

            # normalize features
            img_features = img_features / img_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # Calculate CLIP-score (cosine similarity)
            batch_clip_scores = logit_scale * (img_features * text_features).sum(dim=1)
            clip_scores.extend(batch_clip_scores.cpu().numpy())
    
    average_clip_score = sum(clip_scores) / len(clip_scores)
    
    return average_clip_score


def main(args):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    clip_score = get_clip_score(args, device)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "clip_score.txt")
    with open(output_file, "a") as f:
        f.write(f"Clip score of {args.generation_path} is: {clip_score}\n")

    print(f"Clip score saved to {output_file}: {clip_score}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
