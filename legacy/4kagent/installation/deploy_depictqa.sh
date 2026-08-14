git clone https://github.com/XPixelGroup/DepictQA.git

cp installation/custom_depictqa_scripts/app_eval.py DepictQA/src/
cp installation/custom_depictqa_scripts/app_comp.py DepictQA/src/

mkdir -p DepictQA/experiments/4kagent
cp installation/custom_depictqa_scripts/config_eval.yaml DepictQA/experiments/4kagent/
cp installation/custom_depictqa_scripts/config_comp.yaml DepictQA/experiments/4kagent/

mkdir -p DepictQA/weights/delta

# download checkpoint
# wget -O DepictQA/weights/ViT-L-14.pt "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
# huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir DepictQA/weights/vicuna-7b-v1.5
# wget -O DepictQA/weights/DQ495K.pt "https://huggingface.co/zhiyuanyou/DepictQA2-DQ495K/resolve/main/ckpt.pt"
# gdown --output DepictQA/weights/some_file_name.ext 1o-PN1iXctWl62Tdb8fZs1eD1Ehv6HBMh
