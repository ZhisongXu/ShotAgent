"""
Copy images from experiment result folders to a single destination folder.

Usage examples:
    python image_export.py \
        --src /path/to/result/folder \
        --dst /path/to/result/folder/Images
"""

import os
import shutil
import argparse


def save_images(src_folder, dest_folder):
    """
    Copy result.png from each subfolder in src_folder into dest_folder,
    renaming them to <image_id>.png (where image_id is the prefix before '-').
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    img_num = 0

    result_folders = os.listdir(src_folder)
    print("result_folders:", result_folders)

    for result_folder in result_folders:
        image_name = result_folder.split('-')[0]
        print("image_name:", image_name)

        src_image_path = os.path.join(src_folder, result_folder, "result.png")
        print("src_img_path:", src_image_path)

        if os.path.exists(src_image_path):
            dest_image_path = os.path.join(dest_folder, f"{image_name}.png")
            shutil.copy(src_image_path, dest_image_path)
            img_num += 1

    print(f"Copied {img_num} images")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Copy result.png images from subfolders into one destination folder."
    )
    parser.add_argument(
        "--src",
        "-s",
        required=True,
        help="Source folder containing result subfolders.",
    )
    parser.add_argument(
        "--dst",
        "-d",
        required=True,
        help="Destination folder where collected images will be saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isdir(args.src):
        raise SystemExit(f"Error: source folder does not exist: {args.src}")

    save_images(args.src, args.dst)
