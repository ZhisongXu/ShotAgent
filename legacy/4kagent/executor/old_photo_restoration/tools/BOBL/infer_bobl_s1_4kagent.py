# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import argparse
from pathlib import Path
from subprocess import call

def run_cmd(command):
    try:
        print(f"Running command:\n{command}")
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted.")
        sys.exit(1)

def make_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photo Restoration Pipeline")
    parser.add_argument("--input_folder", type=str, default="./test_images/old", help="Input test image folder")
    parser.add_argument("--output_folder", type=str, default="./output", help="Output folder (absolute path recommended)")
    parser.add_argument("--GPU", type=str, default="0", help="GPU ID(s), e.g., 0 or 0,1")
    parser.add_argument("--checkpoint_name", type=str, default="Setting_9_epoch_100", help="Checkpoint name to use")
    parser.add_argument("--with_scratch", action="store_true", help="Enable scratch detection and restoration")
    parser.add_argument("--HR", action="store_true", help="Enable high-resolution mode")

    opts = parser.parse_args()

    # Resolve paths
    input_folder = Path(opts.input_folder).resolve()
    output_folder = Path(opts.output_folder).resolve()
    make_dir(output_folder)

    gpu_ids = opts.GPU

    # === Stage 1: Overall Restoration ===
    print("Running Stage 1: Overall Restoration")
    stage1_input_dir = input_folder
    stage1_output_dir = output_folder / "stage_1_restore_output"
    make_dir(stage1_output_dir)

    os.chdir("./Global")

    if not opts.with_scratch:
        stage1_command = (
            f"python test.py --test_mode Full --Quality_restore "
            f"--test_input {stage1_input_dir} --outputs_dir {stage1_output_dir} "
            f"--gpu_ids {gpu_ids}"
        )
        run_cmd(stage1_command)

    else:
        # === Scratch + Quality Restoration ===
        mask_dir = stage1_output_dir / "masks"
        input_mask_dir = mask_dir / "input"
        mask_mask_dir = mask_dir / "mask"
        make_dir(mask_dir)

        detection_command = (
            f"python detection.py --test_path {stage1_input_dir} --output_dir {mask_dir} "
            f"--input_size full_size --GPU {gpu_ids}"
        )
        run_cmd(detection_command)
        
        hr_flag = " --HR" if opts.HR else ""
        restoration_command = (
            f"python test.py --Scratch_and_Quality_restore "
            f"--test_input {input_mask_dir} --test_mask {mask_mask_dir} "
            f"--outputs_dir {stage1_output_dir} --gpu_ids {gpu_ids}{hr_flag}"
        )
        run_cmd(restoration_command)

    print("Stage 1 completed.\n")
