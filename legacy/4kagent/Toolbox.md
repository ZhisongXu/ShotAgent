# Methods and Pretrained Models

Methods and pre-trained models used in 4KAgent are summarized as follows:

| Task | Method | Method Repo | Pretrained model from |
|---|---|---|---|
| Brightening | CLAHE | N/A | N/A |
| Brightening | Constant Shift (C = 40) | N/A | N/A |
| Brightening | DiffPlugin | https://github.com/yuhaoliu7456/Diff-Plugin | https://github.com/yuhaoliu7456/Diff-Plugin/tree/main/pre-trained/lowlight |
| Brightening | FourierDiff | https://github.com/aipixel/fourierdiff | https://openai-public.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt |
| Brightening | Gamma Correction | N/A | N/A |
| Brightening | MAXIM | https://github.com/google-research/maxim | https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Enhancement/LOL;tab=objects?prefix=&forceOnObjectsSortingFiltering=false |
| Defocus Deblurring | ConvIR | https://github.com/c-yn/ConvIR | https://drive.google.com/drive/folders/1_5fO2p5xoWO5cUEVoXJ7x3Uhg1AP18FQ -> dpdd-large.pkl |
| Defocus Deblurring | DiffPlugin | https://github.com/yuhaoliu7456/Diff-Plugin | https://github.com/yuhaoliu7456/Diff-Plugin/tree/main/pre-trained/deblur |
| Defocus Deblurring | DRBNet | https://github.com/lingyanruan/DRBNet | https://drive.google.com/file/d/1vGlmev9LdagttXE_nN1gZGVstVTRVQHT/view?usp=sharing -> single_image_defocus_deblurring.pth |
| Defocus Deblurring | IFAN | https://github.com/codeslake/IFAN | https://www.dropbox.com/s/qohrmpr9b1u0syi/checkpoints.zip?dl=1 -> IFAN_44.pytorch |
| Defocus Deblurring | LaKDNet | https://github.com/lingyanruan/LaKDNet | https://lakdnet.mpi-inf.mpg.de/ -> train_on_dpdd_l.pth |
| Defocus Deblurring | Restormer | https://github.com/swz30/Restormer | https://drive.google.com/drive/folders/1bRBG8DG_72AGA6-eRePvChIT5ZO4cwJ4 -> single_image_defocus_deblurring.pth |
| Motion Deblurring | EVSSM | https://github.com/kkkis/EVSSM | https://drive.google.com/drive/folders/1_G0hf1KX_SAvyK41bdGm2q0q63B6wN40 -> net_g_GoPro.pth |
| Motion Deblurring | LaKDNet | https://github.com/lingyanruan/LaKDNet | https://lakdnet.mpi-inf.mpg.de/ -> train_on_realr_l.pth |
| Motion Deblurring | MAXIM | https://github.com/google-research/maxim | https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Deblurring/RealBlur_R;tab=objects?prefix=&forceOnObjectsSortingFiltering=false |
| Motion Deblurring | MPRNet | https://github.com/swz30/MPRNet | https://drive.google.com/file/d/1QwQUVbk6YVOJViCsOKYNykCsdJSVGRtb/view |
| Motion Deblurring | NAFNet | https://github.com/megvii-research/NAFNet | https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view |
| Motion Deblurring | Restormer | https://github.com/swz30/Restormer | https://drive.google.com/drive/folders/1czMyfRTQDX3j3ErByYeZ1PM4GVLbJeGk |
| Motion Deblurring | X-Restormer | https://github.com/Andrew0613/X-Restormer | https://drive.google.com/drive/folders/1C8hEafMx8ivmujzk7l2c_bPhGRu6xBHX |
| Dehazing | DehazeFormer | https://github.com/IDKiro/DehazeFormer | https://drive.google.com/drive/folders/1qnQil_7Dvy-ZdQUVYXt7pW0EFQkpK39B |
| Dehazing | DiffPlugin | https://github.com/yuhaoliu7456/Diff-Plugin | https://github.com/yuhaoliu7456/Diff-Plugin/tree/main/pre-trained/dehaze |
| Dehazing | MAXIM | https://github.com/google-research/maxim | https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Dehazing/SOTS-Outdoor;tab=objects?prefix=&forceOnObjectsSortingFiltering=false |
| Dehazing | RIDCP | https://github.com/RQ-Wu/RIDCP_dehazing | https://pan.baidu.com/share/init?surl=ps9dPmerWyXlLxb6lkHihQ (pwd:huea) |
| Dehazing | X-Restormer | https://github.com/Andrew0613/X-Restormer | https://drive.google.com/drive/folders/1_Cc2_3_P7hU4aTIWoa-lU3Ug70fS0oHd |
| Deraining | DiffPlugin | https://github.com/yuhaoliu7456/Diff-Plugin | https://github.com/yuhaoliu7456/Diff-Plugin/tree/main/pre-trained/derain |
| Deraining | MAXIM | https://github.com/google-research/maxim | https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Deraining/Rain13k;tab=objects?prefix=&forceOnObjectsSortingFiltering=false |
| Deraining | MPRNet | https://github.com/swz30/MPRNet | https://drive.google.com/file/d/103WEJbcat7eTY6doXWeorAB1oMmMnM/view |
| Deraining | Restormer | https://github.com/swz30/Restormer | https://drive.google.com/drive/folders/1ZEDDEW0UkqpWk-N4Li_JUoVGhGXCu_u |
| Deraining | X-Restormer | https://github.com/Andrew0613/X-Restormer | https://drive.google.com/drive/folders/16eCJ7R1CDYhDLqS7KzFUZY_zD1uo2S |
| Denoising | MAXIM | https://github.com/google-research/maxim | https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Denoising/SIDD;tab=objects?prefix=&forceOnObjectsSortingFiltering=false |
| Denoising | MPRNet | https://github.com/swz30/MPRNet | https://drive.google.com/file/d/1LODPl9kYJmxuU98g9dRuRoanE5HyfCsRw/view |
| Denoising | NAFNet | https://github.com/megvii-research/NAFNet | https://drive.google.com/file/d/14ff1hQZJqzMIkAntEpXoWJbU6aIWWR/view |
| Denoising | Restormer | https://github.com/swz30/Restormer | https://drive.google.com/drive/folders/1Qswjny54ZRWa7cA4ap7exiXLBo4uF40 -> real_denoising.pth |
| Denoising | X-Restormer | https://github.com/Andrew0613/X-Restormer | https://drive.google.com/drive/folders/1-vcU7JiJQNSewTbNCS2j9lIKOZ0W89 |
| Denoising | SwinIR (15) | https://github.com/JingyunLiang/SwinIR | https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0 -> 005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth |
| Denoising | SwinIR (50) | https://github.com/JingyunLiang/SwinIR | https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0 -> 005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth |
| JPEG CAR | FBANN (5) | https://github.com/jiaxi-jiang/FBCNN | https://github.com/jiaxi-jiang/FBCNN/releases/tag/v1.0 -> fbcnn_color.pth |
| JPEG CAR | FBANN (90) | https://github.com/jiaxi-jiang/FBCNN | https://github.com/jiaxi-jiang/FBCNN/releases/tag/v1.0 -> fbcnn_color.pth |
| JPEG CAR | FBANN (B) | https://github.com/jiaxi-jiang/FBCNN | https://github.com/jiaxi-jiang/FBCNN/releases/tag/v1.0 -> fbcnn_color.pth |
| JPEG CAR | SwinIR | https://github.com/JingyunLiang/SwinIR | https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0 -> 006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth |
| Super-Resolution | DiffBIR | https://github.com/XPixelGroup/DiffBIR | https://huggingface.co/xql007/diffbir-v2/resolve/main/v2.pt |
| Super-Resolution | DRCT | https://github.com/ning0531/DRCT | https://drive.google.com/drive/folders/1J0JHdSfOePNh96ibgzMJAJPw31ug9z7U -> DRCT-L_X4.pth |
| Super-Resolution | HAT-L | https://github.com/XPixelGroup/HAT | https://drive.google.com/drive/folders/1HpmReFoUqUnBAOQ7yoENuU3uf_m69w0 -> HAT-L_SRx4_ImageNet-pretrain.pth |
| Super-Resolution | HAT-GAN | https://github.com/XPixelGroup/HAT | https://drive.google.com/drive/folders/1HpmReFoUqUnBAOQ7yoENuU3uf_m69w0 -> Real_HAT_GAN_sharper.pth |
| Super-Resolution | HMA | https://github.com/Korouoruw/HMA | https://drive.google.com/drive/folders/13Bxt_BXtVWQ0crgL7lo6rYZAUY1v3zi -> HMA_SRx4_pretrain.pth |
| Super-Resolution | OSEDiff | https://github.com/cswry/OSEDiff | https://github.com/cswry/OSEDiff/tree/main/prest/models |
| Super-Resolution | PiSA-SR | https://github.com/csslc/PiSA-SR | https://drive.google.com/drive/folders/1eljWWMds9xJwseL2SEoJvXQBiFxWds |
| Super-Resolution | SwinIR | https://github.com/JingyunLiang/SwinIR | https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0 -> 003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth |
| Super-Resolution | SwinIR (Real-ISR) | https://github.com/JingyunLiang/SwinIR | https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0 -> 003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth |
| Super-Resolution | SwinFIR | https://github.com/ZdAeng/SwinFIR | https://drive.google.com/drive/folders/9dQAXbISB7zEy2UJK7AU1W6EJloK -> SwinFIR_SRx4.pth |
| Super-Resolution | X-Restormer | https://github.com/Andrew0613/X-Restormer | https://drive.google.com/drive/folders/1GiK8cfRdcjAPPN3Qt6b7AKveBAH_AS |
| Face-Restoration | GFPGAN | https://github.com/TencentARC/GFPGAN | https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth |
| Face-Restoration | CodeFormer | https://github.com/sczhou/CodeFormer | https://drive.google.com/drive/folders/1S_wQNBNYjbHFtD95qb4YMv6plfo5iuU6QS |
| Face-Restoration | DiffFace | https://github.com/zsyOAOA/DiffFace | https://github.com/zsyOAOA/DiffFace/releases/tag/V1.0 |
| Old Photo Restoration | BOBL | https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life | https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/tag/v1.0 |
| Human Preference Evaluation | HPSv2 | https://github.com/tgxs002/HPSv2 | https://github.com/tgxs002/HPSv2/blob/master/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz |


We sincerely acknowledge these open-source codes and pre-trained models.