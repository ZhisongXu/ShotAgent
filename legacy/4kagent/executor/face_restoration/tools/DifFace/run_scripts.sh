# Face Restoration
python inference_difface.py -i [image folder/image path] -o [result folder] --task restoration --eta 0.5 --aligned --use_fp16

# Whole Image Restoration
python inference_difface.py -i [image folder/image path] -o [result folder] --task restoration --eta 0.5 --use_fp16




CUDA_VISIBLE_DEVICES=0 python inference_difface.py -i /path_to_Face_Images/WebPhoto-Test/WebPhoto-Test_128_x4 -o /path_to_output --task restoration --eta 0.5 --aligned --use_fp16