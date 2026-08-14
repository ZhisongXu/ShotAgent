#!/bin/bash

# Get the project root directory (the parent directory of 'installation')
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ln -s "$PROJECT_ROOT/executor/dehazing/tools/maxim" "$PROJECT_ROOT/executor/brightening/tools/"
ln -s "$PROJECT_ROOT/executor/defocus_deblurring/tools/AutoDIR" "$PROJECT_ROOT/executor/brightening/tools/"
ln -s "$PROJECT_ROOT/executor/defocus_deblurring/tools/Diff-Plugin" "$PROJECT_ROOT/executor/brightening/tools/"

ln -s "$PROJECT_ROOT/executor/defocus_deblurring/tools/Diff-Plugin" "$PROJECT_ROOT/executor/dehazing/tools/"
ln -s "$PROJECT_ROOT/executor/defocus_deblurring/tools/AutoDIR" "$PROJECT_ROOT/executor/dehazing/tools/"

ln -s "$PROJECT_ROOT/executor/dehazing/tools/maxim" "$PROJECT_ROOT/executor/denoising/tools/"
ln -s "$PROJECT_ROOT/executor/dehazing/tools/X-Restormer" "$PROJECT_ROOT/executor/denoising/tools/"
ln -s "$PROJECT_ROOT/executor/defocus_deblurring/tools/Restormer" "$PROJECT_ROOT/executor/denoising/tools/"

ln -s "$PROJECT_ROOT/executor/dehazing/tools/maxim" "$PROJECT_ROOT/executor/deraining/tools/"
ln -s "$PROJECT_ROOT/executor/dehazing/tools/X-Restormer" "$PROJECT_ROOT/executor/deraining/tools/"
ln -s "$PROJECT_ROOT/executor/denoising/tools/MPRNet" "$PROJECT_ROOT/executor/deraining/tools/"
ln -s "$PROJECT_ROOT/executor/defocus_deblurring/tools/Restormer" "$PROJECT_ROOT/executor/deraining/tools/"
ln -s "$PROJECT_ROOT/executor/defocus_deblurring/tools/Diff-Plugin" "$PROJECT_ROOT/executor/deraining/tools/"

ln -s "$PROJECT_ROOT/executor/denoising/tools/SwinIR" "$PROJECT_ROOT/executor/jpeg_compression_artifact_removal/tools/"

ln -s "$PROJECT_ROOT/executor/dehazing/tools/maxim" "$PROJECT_ROOT/executor/motion_deblurring/tools/"
ln -s "$PROJECT_ROOT/executor/dehazing/tools/X-Restormer" "$PROJECT_ROOT/executor/motion_deblurring/tools/"
ln -s "$PROJECT_ROOT/executor/defocus_deblurring/tools/Restormer" "$PROJECT_ROOT/executor/motion_deblurring/tools/"
ln -s "$PROJECT_ROOT/executor/denoising/tools/MPRNet" "$PROJECT_ROOT/executor/motion_deblurring/tools/"
ln -s "$PROJECT_ROOT/executor/denoising/tools/NAFNet" "$PROJECT_ROOT/executor/motion_deblurring/tools/"

ln -s "$PROJECT_ROOT/executor/denoising/tools/SwinIR" "$PROJECT_ROOT/executor/super_resolution/tools/"
ln -s "$PROJECT_ROOT/executor/denoising/tools/X-Restormer" "$PROJECT_ROOT/executor/super_resolution/tools/"
ln -s "$PROJECT_ROOT/executor/denoising/tools/NAFNet" "$PROJECT_ROOT/executor/super_resolution/tools/"
