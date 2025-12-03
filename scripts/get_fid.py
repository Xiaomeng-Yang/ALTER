import argparse
import os
import logging
from cleanfid import fid

def parse_args():
    parser = argparse.ArgumentParser(description="Compute FID using cleanfid.")
    parser.add_argument('--output_dir', type=str, required=True,
                         help="Directory to save the FID score.")
    parser.add_argument('--generation_path', type=str, required=True,
                         help="Path to the generated images.")
    parser.add_argument('--dataset', type=str, default="cocoval17",
                         help="Name of the dataset to compare against.")
    parser.add_argument('--dataset_path', type=str, default="/work/yanzhi_group/yang.xiaome/dataset/T2I/mscoco/val2017_npy256",
                         help="Path to the real dataset images.")
    parser.add_argument('--fid_mode', type=str, default='legacy_pytorch',
                         help="Mode for FID computation (e.g., legacy_pytorch or tensorflow).")

    args = parser.parse_args()
    return args


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logging.info(f"Checking FID stats for dataset: {args.dataset}")
    if not fid.test_stats_exists(args.dataset, mode=args.fid_mode):
        logging.info(f"Stats not found. Creating custom stats for dataset: {args.dataset}")
        fid.make_custom_stats(args.dataset, args.dataset_path, mode=args.fid_mode)
    
    fid_value = fid.compute_fid(args.generation_path, dataset_name=args.dataset, mode=args.fid_mode, dataset_split="custom")
    logging.info(f"FID computed: {fid_value}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "fid.txt")
    with open(output_file, "a") as f:
        f.write(f"FID of {args.generation_path} is: {fid_value}\n")

    logging.info(f"FID value saved to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
