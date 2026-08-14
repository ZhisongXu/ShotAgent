CUDA_VISIBLE_DEVICES=0 python inference/inference_swinfir.py \
    --input="/path_to_classicsr_dataset/Set5/LR_bicubic/X4" \
    --output="/path_to_swinfir_output/Set5" \
    --model_path="/path_to_swinfir_pretrained_models/SwinFIR_SRx4.pth" \
    --scale=4

CUDA_VISIBLE_DEVICES=0 python inference/inference_swinfir.py \
    --input="/path_to_classicsr_dataset/Set14/LR_bicubic/X4" \
    --output="/path_to_swinfir_output/Set14" \
    --model_path="/path_to_swinfir_pretrained_models/SwinFIR_SRx4.pth" \
    --scale=4

CUDA_VISIBLE_DEVICES=0 python inference/inference_swinfir.py \
    --input="/path_to_classicsr_dataset/B100/LR_bicubic/X4" \
    --output="/path_to_swinfir_output/B100" \
    --model_path="/path_to_swinfir_pretrained_models/SwinFIR_SRx4.pth" \
    --scale=4

CUDA_VISIBLE_DEVICES=0 python inference/inference_swinfir.py \
    --input="/path_to_classicsr_dataset/Urban100/LR_bicubic/X4" \
    --output="/path_to_swinfir_output/Urban100" \
    --model_path="/path_to_swinfir_pretrained_models/SwinFIR_SRx4.pth" \
    --scale=4

CUDA_VISIBLE_DEVICES=0 python inference/inference_swinfir.py \
    --input="/path_to_classicsr_dataset/Manga109/LR_bicubic/X4" \
    --output="/path_to_swinfir_output/Manga109" \
    --model_path="/path_to_swinfir_pretrained_models/SwinFIR_SRx4.pth" \
    --scale=4