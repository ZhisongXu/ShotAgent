# Intsallation

## Hardware requirements

`Storage:` >50 GB disk storage (Code + pretrained models of restoration methods + VLM / LLM setup).

`GPU:` >= 2 GPUs (>=24 GB VRAM) for multi-GPU deployment.

## Enviroment Setting

```bash
# 4kagent base environment
conda create -n 4kagent python=3.10 -y
conda activate 4kagent
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r installation/requirements.txt
pip install --upgrade transformers==4.49.0
conda deactivate

# 4kagent dehaze environment [specific to RIDCP]
conda create -n 4kagent_dehaze python=3.8 -y
conda activate 4kagent_dehaze
cd executor/dehazing/tools/RIDCP_dehazing
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
BASICSR_EXT=True python setup.py develop
cd ../../../..
conda deactivate

# 4kagent super-resolution environment [specific to OSEDiff, PISASR]
conda create -n 4kagent_sr python=3.10 -y
conda activate 4kagent_sr
cd executor/super_resolution/tools/OSEDiff
pip install -r requirements.txt
cd ../../../..
conda deactivate

# 4kagent other tools environment   # executor/tool.py L71
conda create -n 4kagent_spec_tools python=3.8 -y
conda activate 4kagent_spec_tools
pip install -r installation/spec_tools_requirements.txt
pip install mamba-ssm
cd executor/dehazing/tools/maxim
pip install .
cd ../../../..
cd executor/denoising/tools/NAFNet
python setup.py develop --no_cuda_ext
cd ../../../..
conda deactivate

# Back to main env
conda activate 4kagent
```


## Deploy Tools & Perception Models

### Create Symbolic Links

To avoid redundant storage of shared model repositories across different degradation types, we use symbolic links:

```bash
bash installation/tools_sym_links.sh
```

### Download pretrained checkpoints

We provide required pretrained checkpoints via [huggingface](https://huggingface.co/YSZuo/4KAgent-Toolbox-Pretrained-Models) and [Google Drive](https://drive.google.com/file/d/1cX1n_zFFteX6_1NeVtoitUi-A_azl7jD/view?usp=drive_link). For google drive, you can download them using gdown:  

```bash
pip install gdown
python -m gdown 1cX1n_zFFteX6_1NeVtoitUi-A_azl7jD
tar -xzvf 4KAgent_toolbox_pretrained_ckpts.tar.gz
rm -rf 4KAgent_toolbox_pretrained_ckpts.tar.gz

# download the landmark detection model for old-photo-restoration
cp pretrained_ckpts/BOBL/Face_Detection/shape_predictor_68_face_landmarks.dat executor/old_photo_restoration/tools/BOBL/Face_Detection/

# put bpe_simple_vocab_16e6.txt.gz
cp pretrained_ckpts/hpsv2/bpe_simple_vocab_16e6.txt.gz utils/clib_fiqa/model/
cp pretrained_ckpts/hpsv2/bpe_simple_vocab_16e6.txt.gz $(python -c "import hpsv2, os; print(os.path.join(os.path.dirname(hpsv2.__file__), 'src/open_clip'))")

# 256x256_diffusion_uncond.pt in FourierDiff # This will also download automatically.
cp pretrained_ckpts/FourierDiff/ckpts executor/brightening/tools/FourierDiff/guided_diffusion/
```
After downloading, unzip it in the project root directory.

### Verify Tool Setup

To verify that all restoration tools are properly installed and functioning, run the test below. This test confirms that each restoration model can run correctly in your environment.

```bash
CUDA_VISIBLE_DEVICES=0 python -m test_tool.test_tool
```

### Deploy [DepictQA](https://github.com/XPixelGroup/DepictQA)

+ Run `sh installation/deploy_depictqa.sh` to prepare the code, which is adapted from the official repo linked above.
+ Set up the environment according to the official repo.
+ Download the weights.
    + Download the pre-trained ViT from [this link](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) and put it in `DepictQA/weights/`.
    + Download the pre-trained Vicuna from [this link](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main) and put it in `DepictQA/weights/`.
    + Download the delta weights of DepictQA-Wild from [this link](https://huggingface.co/zhiyuanyou/DepictQA2-DQ495K/tree/main), rename it to `DQ495K.pt`, and put it in `DepictQA/weights/delta/`.
    + Download the delta weights fine-tuned from DepictQA-Wild from [this link](https://drive.google.com/file/d/1o-PN1iXctWl62Tdb8fZs1eD1Ehv6HBMh/view?usp=drive_link) and put it in `DepictQA/weights/delta/`.

    The structure of `DepictQA/weights` should look like this:
    ```
    DepictQA/weights/
    ├── ViT-L-14.pt
    ├── vicuna-7b-v1.5/
    │   └── ...
    └── delta/
        ├── DQ495K.pt
        └── degra_eval.pt
    ```

    ```bash
    sh installation/deploy_depictqa.sh
    cd ./DepictQA
    conda create -n depictqa python=3.10 -y
    conda activate depictqa
    pip install -r requirements.txt
    pip install "huggingface_hub[cli]" gdown

    cd ./weights/
    wget "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
    huggingface-cli download "lmsys/vicuna-7b-v1.5" --local-dir "./vicuna-7b-v1.5"
    huggingface-cli download "zhiyuanyou/DepictQA2-DQ495K" --local-dir "./DepictQA2-DQ495K"
    mv ./DepictQA2-DQ495K/ckpt.pt ./delta/DQ495K.pt
    gdown 1o-PN1iXctWl62Tdb8fZs1eD1Ehv6HBMh -O ./delta
    mv ./delta/deltao_fk_5sj.part ./delta/degra_eval.pt
    ```

+ To launch DepictQA for evaluation using the DepictQA
    ```bash
    cd ./DepictQA
    conda activate depictqa
    CUDA_VISIBLE_DEVICES=0 python src/app_eval.py
    CUDA_VISIBLE_DEVICES=0 python src/app_comp.py
    ```