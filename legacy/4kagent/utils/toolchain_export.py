"""
Extract toolchain information from workflow logs.

Usage examples:
    python toolchain_export.py \
        --base-dir /Path/to/Result/Folder \
        --output-file /Path/to/Result/Folder/Results_toolchain.txt
"""

import os
import argparse


def extract_toolchain_info(base_dir, output_file):
    """
    Traverse directories under `base_dir`, read each `logs/workflow.log`, and
    extract the toolchain string from the second-to-last line when it contains
    "Restoration result:". The function writes results to `output_file` in the
    format "<image_id>: <toolchain>" and also returns the results as a dict.
    """
    result = {}

    # List and sort entries in base_dir
    folders = sorted(os.listdir(base_dir))

    # Iterate through each entry and process directories only
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)

        if os.path.isdir(folder_path):
            # Construct the path to the workflow log inside the logs subdirectory
            logs_path = os.path.join(folder_path, "logs", "workflow.log")

            # If the log exists, open and process it
            if os.path.exists(logs_path):
                try:
                    with open(logs_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-2].strip()
                            if "Restoration result:" in last_line:
                                # Extract the toolchain part after the marker
                                toolchain = last_line.split("Restoration result:")[-1].strip()
                                # Extract original image id from folder name (part before first '-')
                                image_id = folder.split('-')[0]
                                result[image_id] = toolchain
                except Exception as e:
                    # Preserve original behavior of printing errors to stdout
                    print(f"Error reading {logs_path}: {e}")

    # Write the collected results into the output text file
    with open(output_file, "w", encoding="utf-8") as f:
        for img_id, toolchain in result.items():
            f.write(f"{img_id}: {toolchain}\n")

    return result


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Extract toolchain info from workflow.log files under folders in base directory."
    )
    parser.add_argument(
        "--base-dir",
        "-b",
        required=True,
        help=f"Base directory containing result folders",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        required=True,
        help=f"Output text file to write results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Ensure base directory exists before calling the main function
    if not os.path.isdir(args.base_dir):
        raise SystemExit(f"Error: base directory does not exist: {args.base_dir}")

    info = extract_toolchain_info(
        base_dir=args.base_dir,
        output_file=args.output_file,
    )

    # Print results to stdout
    for img_id, toolchain in info.items():
        print(f"{img_id}: {toolchain}")
