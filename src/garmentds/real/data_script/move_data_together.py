import os
import shutil
from collections import defaultdict


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", "-f", type=str, required=True, help="Folder name of the experiment")
    parser.add_argument("--garment_name", "-g", type=str, default="tshirt")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    cnt = defaultdict(int)
    total_cnt = defaultdict(int)
    folder_name = args.folder
    src_dir = f"outputs/{folder_name}"
    dst_dir = f"data/keypoints/{folder_name}"

    for dirpath, dirnames, filenames in sorted(os.walk(src_dir)):
        for filename in sorted(filenames):
            if filename.endswith('.png') and args.garment_name in dirpath:
                clothes_name = os.path.basename(dirpath)
                total_cnt[clothes_name] += 1
    
    for dirpath, dirnames, filenames in sorted(os.walk(src_dir)):
        for filename in sorted(filenames):
            if filename.endswith('.png') and args.garment_name in dirpath:
                clothes_name = os.path.basename(dirpath)
                src_path = os.path.join(dirpath, filename)
                dst_path = os.path.join(dst_dir, clothes_name, f"{str(cnt[clothes_name]).zfill(len(str(total_cnt[clothes_name] - 1)))}_color.png")
                if os.path.exists(dst_path):
                    s = input(f"path {dst_path} exists\npress y to continue")
                    if s != "y":
                        continue
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy(src_path, dst_path)
                shutil.copy(src_path.replace("color.png", "depth.npy"), dst_path.replace("color.png", "depth.npy"))
                cnt[clothes_name] += 1

if __name__ == '__main__':
    main()