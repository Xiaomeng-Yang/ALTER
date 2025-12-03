import argparse
import os
import json
import numpy as np
import cv2

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="test script.")
    parser.add_argument('--img_dir', type=str, default="/data/mscoco/val2017")
    parser.add_argument('--caption_path', type=str, default="/data/mscoco/annotations/captions_val2017.json")
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
    args = parser.parse_args()
    return args


def main(args):
    os.makedirs(args.output_path, exist_ok=True)

    print("=====> Load MS-COCO dataset <=====")
    caption_file = json.load(open(args.caption_path))
    images = []
    texts = []
    image_ids = {}
    for temp in caption_file['annotations']:
        # process the several captions for one image
        cur_image_id = temp['image_id']
        if cur_image_id in image_ids:
            image_ids[cur_image_id] += 1
        else:
            image_ids[cur_image_id] = 0

        anno_index = image_ids[cur_image_id]
        # Build the final .npy filename you will save to:
        filename_npy = f"{cur_image_id:012d}_{anno_index}.npy"
        out_file = os.path.join(args.output_path, filename_npy)

        # If it already exists in output_path, we skip it
        if os.path.exists(out_file):
            # Already generated => skip
            continue

        img_path = os.path.join(args.img_dir, "%012d_%d.jpg" % (cur_image_id, image_ids[cur_image_id]))
        caption = temp['caption']
        images.append(img_path)
        texts.append(caption)

    for i in range(len(texts)):
        img_path = images[i]
        img = Image.open(img_path[:-6]+'.jpg').convert("RGB")
        data = np.asarray(img)
        data = cv2.resize(data, (256, 256))
        img_name = img_path.split("/")[-1]
        if img_name.endswith(".jpg"):
            img_name = img_name[:-4]
        output_path = os.path.join(args.output_path, f"{img_name}.npy")
        np.save(output_path, data)


if __name__ == "__main__":
    args = parse_args()
    main(args)

