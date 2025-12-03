import json
import os
import pandas as pd
from datasets import Dataset


def save_coco_captions(annotations_file):
    # annotations file's name is something like 'annotations/captions_val2014_30k.json'
    split_name = os.path.basename(annotations_file)[len('captions_'):-len('.json')]
    captions_file = json.load(open(annotations_file))
    captions_dir = os.path.dirname(annotations_file)
    save_dir = os.path.join(captions_dir, 'clip-captions')
    os.makedirs(save_dir, exist_ok=True)
    for capt in captions_file['annotations']:
        if '2014' in annotations_file:
            image_id = f"COCO_{split_name}_%012d" % capt['image_id']
        else:
            image_id = "%012d" % capt['image_id']

        caption = capt['caption']
        with open(os.path.join(save_dir, image_id + '.txt'), 'w') as f:
            f.write(caption)


def process_cc3m(img_dir, caption_path):
    captions = pd.read_csv(caption_path, sep="\t", header=None, names=["caption", "link"], dtype={"caption": str, "link": str})

    images = os.listdir(img_dir)
    images = [os.path.join(img_dir, image) for image in images]

    sorted_pairs = sorted([(int(os.path.basename(image).split("_")[0]), image) for image in images])
    sorted_image_indices, sorted_images = zip(*sorted_pairs)
    sorted_image_indices = list(sorted_image_indices)
    sorted_images = list(sorted_images)

    captions = captions.iloc[sorted_image_indices].caption.values.tolist()
    dataset = Dataset.from_dict({"image": sorted_images, "text": captions})
    return dataset


def save_cc3m_captions(data_dir, caption_path):
    dataset = process_cc3m(data_dir, caption_path)
    save_dir = os.path.join(data_dir, 'clip-captions')
    os.makedirs(save_dir, exist_ok=True)
    print(len(dataset))
    for sample in dataset:
        image_id = sample['image'][:-4]
        caption = sample['text']
        with open(os.path.join(save_dir, image_id + '.txt'), 'w') as f:
            f.write(caption)


if __name__ == '__main__':
    # save_coco_captions('/work/yanzhi_group/yang.xiaome/dataset/T2I/coco/annotations/captions_val2014_30k.json')
    save_cc3m_captions("/work/yanzhi_group/yang.xiaome/dataset/T2I/CC3M/validation_256npy_new/",
        "/work/yanzhi_group/yang.xiaome/dataset/T2I/CC3M/Validation_GCC-1.1.0-Validation.tsv")
