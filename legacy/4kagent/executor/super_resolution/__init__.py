import os
from shutil import rmtree
from pathlib import Path

from ..tool import Tool
from ..multitask_tools import *


__all__ = ['sr_toolbox']
project_root = Path(__file__).resolve().parents[2]


class HATPSNR(BasicSRModel):
    """[Activating More Pixels in Image Super-Resolution Transformer (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Activating_More_Pixels_in_Image_Super-Resolution_Transformer_CVPR_2023_paper.pdf)"""

    def __init__(self):
        super().__init__(
            tool_name="hat_psnr",
            subtask="super_resolution",
            work_dir="HAT",
            script_rel_path=Path("hat")/'infer_hat_4kagent.py'
        )
    
    def _update_pretrained_ckpt(self, cfg: dict):
        ckpt_name = cfg['path']['pretrain_network_g']
        cfg['path']['pretrain_network_g'] = str(project_root / f"pretrained_ckpts/{self.work_dir_name}/{ckpt_name}")
        

class HATPSNR_2x(BasicSRModel):
    """[Activating More Pixels in Image Super-Resolution Transformer (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Activating_More_Pixels_in_Image_Super-Resolution_Transformer_CVPR_2023_paper.pdf)"""

    def __init__(self):
        super().__init__(
            tool_name="hat_psnr_2x",
            subtask="super_resolution",
            work_dir="HAT",
            script_rel_path=Path("hat")/'infer_hat_4kagent.py'
        )
        
    def _update_pretrained_ckpt(self, cfg: dict):
        ckpt_name = cfg['path']['pretrain_network_g']
        cfg['path']['pretrain_network_g'] = str(project_root / f"pretrained_ckpts/{self.work_dir_name}/{ckpt_name}")


class HATGAN(BasicSRModel):
    """[Activating More Pixels in Image Super-Resolution Transformer (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Activating_More_Pixels_in_Image_Super-Resolution_Transformer_CVPR_2023_paper.pdf)"""

    def __init__(self):
        super().__init__(
            tool_name="hat_gan",
            subtask="super_resolution",
            work_dir="HAT",
            script_rel_path=Path("hat")/'infer_hat_4kagent.py'
        )

    def _update_pretrained_ckpt(self, cfg: dict):
        ckpt_name = cfg['path']['pretrain_network_g']
        cfg['path']['pretrain_network_g'] = str(project_root / f"pretrained_ckpts/{self.work_dir_name}/{ckpt_name}")
        

class SwinFIR(Tool):
    """One-Step Effective Diffusion Network for Real-World Image Super-Resolution [NeurIPS2024](https://arxiv.org/abs/2406.08177)"""
    def __init__(self):
        super().__init__(
            tool_name="swinfir",
            subtask="super_resolution",
            work_dir="SwinFIR",
            script_rel_path="infer_swinfir_4kagent.py"
        )
    
    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input", self.input_dir,
            "--output", self.output_dir,
            "--model_path", str(project_root / "pretrained_ckpts/SwinFIR/SwinFIR_SRx4.pth"),
            "--scale", "4"
        ]


class SwinFIR_2x(Tool):
    """One-Step Effective Diffusion Network for Real-World Image Super-Resolution [NeurIPS2024](https://arxiv.org/abs/2406.08177)"""
    def __init__(self):
        super().__init__(
            tool_name="swinfir_2x",
            subtask="super_resolution",
            work_dir="SwinFIR",
            script_rel_path="infer_swinfir_4kagent.py"
        )
    
    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input", self.input_dir,
            "--output", self.output_dir,
            "--model_path", str(project_root / "pretrained_ckpts/SwinFIR/SwinFIR_SRx2.pth"),
            "--scale", "2"
        ]


class DRCT(Tool):
    """[Activating More Pixels in Image Super-Resolution Transformer (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Activating_More_Pixels_in_Image_Super-Resolution_Transformer_CVPR_2023_paper.pdf)"""
    def __init__(self):
        super().__init__(
            tool_name="drct",
            subtask="super_resolution",
            work_dir="DRCT",
            script_rel_path="infer_drct_4kagent.py"
        )
    
    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input", self.input_dir,
            "--output", self.output_dir,
            "--model_path", str(project_root / "pretrained_ckpts/DRCT/DRCT-L_X4.pth"),
            "--scale", "4",
            "--tile", "512"
        ]
        
    def _update_pretrained_ckpt(self, cfg: dict):
        ckpt_name = cfg['path']['pretrain_network_g']
        cfg['path']['pretrain_network_g'] = str(project_root / f"pretrained_ckpts/{self.work_dir_name}/{ckpt_name}")



class DiffBIR(Tool):
    """[DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior (ECCV 2024)](https://arxiv.org/abs/2308.15070)"""    

    def __init__(self):
        super().__init__(
            tool_name="diffbir",
            subtask="super_resolution",
            work_dir="DiffBIR",
            script_rel_path="infer_diffbir_4kagent.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--task", "sr",
            "--input", self.input_dir,
            "--pos_prompt", "",
            "--neg_prompt", "low quality, blurry, low-resolution, noisy, unsharp, weird textures",
            "--sampler", "spaced",
            "--version", "v2",
            "--steps", "50",
            "--upscale", "4",
            "--cfg_scale", "4",
            "--captioner", "none",
            "--output", self.output_dir,
            "--device", "cuda",
            "--cleaner_tiled",
            "--vae_encoder_tiled",
            "--vae_decoder_tiled",
            "--cldm_tiled"
        ]


class DiffBIR_2x(Tool):
    """[DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior (ECCV 2024)](https://arxiv.org/abs/2308.15070)"""    

    def __init__(self):
        super().__init__(
            tool_name="diffbir_2x",
            subtask="super_resolution",
            work_dir="DiffBIR",
            script_rel_path="infer_diffbir_4kagent.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--task", "sr",
            "--input", self.input_dir,
            "--pos_prompt", "",
            "--neg_prompt", "low quality, blurry, low-resolution, noisy, unsharp, weird textures",
            "--sampler", "spaced",
            "--version", "v2",
            "--steps", "50",
            "--upscale", "2",
            "--cfg_scale", "4",
            "--captioner", "none",
            "--output", self.output_dir,
            "--device", "cuda",
            "--cleaner_tiled",
            "--vae_encoder_tiled",
            "--vae_decoder_tiled",
            "--cldm_tiled"
        ]
        

class DiffBIR_16x(Tool):
    """[DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior (ECCV 2024)](https://arxiv.org/abs/2308.15070)"""    

    def __init__(self):
        super().__init__(
            tool_name="diffbir_16x",
            subtask="super_resolution",
            work_dir="DiffBIR",
            script_rel_path="infer_diffbir_4kagent.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--task", "sr",
            "--input", self.input_dir,
            "--pos_prompt", "",
            "--neg_prompt", "low quality, blurry, low-resolution, noisy, unsharp, weird textures",
            "--sampler", "spaced",
            "--version", "v2",
            "--steps", "50",
            "--upscale", "16",
            "--cfg_scale", "4",
            "--captioner", "none",
            "--output", self.output_dir,
            "--device", "cuda",
            "--cleaner_tiled",
            "--vae_encoder_tiled",
            "--vae_decoder_tiled",
            "--cldm_tiled"
        ]


class OSEDiff(Tool):
    """One-Step Effective Diffusion Network for Real-World Image Super-Resolution [NeurIPS2024](https://arxiv.org/abs/2406.08177)"""
    def __init__(self):
        super().__init__(
            tool_name="osediff",
            subtask="super_resolution",
            work_dir="OSEDiff",
            script_rel_path="infer_osediff_4kagent.py"
        )
    
    def _get_cmd_opts(self) -> list[str]:
        return [
            "--i", self.input_dir,
            "--output_dir", self.output_dir,
            "--osediff_path", str(project_root / "executor/super_resolution/tools/OSEDiff/preset/models/osediff.pkl"),
            "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-2-1-base",
            "--ram_ft_path", str(project_root / "pretrained_ckpts/OSEDiff/DAPE/DAPE.pth"),
            "--ram_path", str(project_root / "pretrained_ckpts/OSEDiff/RAM/ram_swin_large_14m.pth"),
            "--vae_decoder_tiled_size", "170"
        ]


class OSEDiff_2x(Tool):
    """One-Step Effective Diffusion Network for Real-World Image Super-Resolution [NeurIPS2024](https://arxiv.org/abs/2406.08177)"""
    def __init__(self):
        super().__init__(
            tool_name="osediff_2x",
            subtask="super_resolution",
            work_dir="OSEDiff",
            script_rel_path="infer_osediff_4kagent.py"
        )
    
    def _get_cmd_opts(self) -> list[str]:
        return [
            "--i", self.input_dir,
            "--output_dir", self.output_dir,
            "--upscale", "2",
            "--osediff_path", str(project_root / "executor/super_resolution/tools/OSEDiff/preset/models/osediff.pkl"),
            "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-2-1-base",
            "--ram_ft_path", str(project_root / "pretrained_ckpts/OSEDiff/DAPE/DAPE.pth"),
            "--ram_path", str(project_root / "pretrained_ckpts/OSEDiff/RAM/ram_swin_large_14m.pth"),
            "--vae_decoder_tiled_size", "170"
        ]
        
        
class OSEDiff_16x(Tool):
    """One-Step Effective Diffusion Network for Real-World Image Super-Resolution [NeurIPS2024](https://arxiv.org/abs/2406.08177)"""
    def __init__(self):
        super().__init__(
            tool_name="osediff_16x",
            subtask="super_resolution",
            work_dir="OSEDiff",
            script_rel_path="infer_osediff_4kagent.py"
        )
    
    def _get_cmd_opts(self) -> list[str]:
        return [
            "--i", self.input_dir,
            "--output_dir", self.output_dir,
            "--upscale", "16",
            "--osediff_path", str(project_root / "executor/super_resolution/tools/OSEDiff/preset/models/osediff.pkl"),
            "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-2-1-base",
            "--ram_ft_path", str(project_root / "pretrained_ckpts/OSEDiff/DAPE/DAPE.pth"),
            "--ram_path", str(project_root / "pretrained_ckpts/OSEDiff/RAM/ram_swin_large_14m.pth"),
            "--vae_decoder_tiled_size", "170"
        ]


class PISASR(Tool):
    """One-Step Effective Diffusion Network for Real-World Image Super-Resolution [NeurIPS2024](https://arxiv.org/abs/2406.08177)"""
    def __init__(self):
        super().__init__(
            tool_name="pisasr",
            subtask="super_resolution",
            work_dir="PiSA-SR",
            script_rel_path="infer_pisasr_4kagent.py"
        )
    
    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_image", self.input_dir,
            "--output_dir", self.output_dir,
            "--pretrained_path", str(project_root / "pretrained_ckpts/PiSA-SR/pisa_sr.pkl"),
            "--pretrained_model_path", "stabilityai/stable-diffusion-2-1-base",
            "--vae_decoder_tiled_size", "170",
            "--default"
        ]
    

class PISASR_2x(Tool):
    """One-Step Effective Diffusion Network for Real-World Image Super-Resolution [NeurIPS2024](https://arxiv.org/abs/2406.08177)"""
    def __init__(self):
        super().__init__(
            tool_name="pisasr_2x",
            subtask="super_resolution",
            work_dir="PiSA-SR",
            script_rel_path="infer_pisasr_4kagent.py"
        )
    
    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_image", self.input_dir,
            "--output_dir", self.output_dir,
            "--upscale", "2",
            "--pretrained_path", str(project_root / "pretrained_ckpts/PiSA-SR/pisa_sr.pkl"),
            "--pretrained_model_path", "stabilityai/stable-diffusion-2-1-base",
            "--vae_decoder_tiled_size", "170",
            "--default"
        ]


class PISASR_16x(Tool):
    """One-Step Effective Diffusion Network for Real-World Image Super-Resolution [NeurIPS2024](https://arxiv.org/abs/2406.08177)"""
    def __init__(self):
        super().__init__(
            tool_name="pisasr_16x",
            subtask="super_resolution",
            work_dir="PiSA-SR",
            script_rel_path="infer_pisasr_4kagent.py"
        )
    
    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_image", self.input_dir,
            "--output_dir", self.output_dir,
            "--upscale", "16",
            "--pretrained_path", str(project_root / "pretrained_ckpts/PiSA-SR/pisa_sr.pkl"),
            "--pretrained_model_path", "stabilityai/stable-diffusion-2-1-base",
            "--vae_decoder_tiled_size", "170",
            "--default"
        ]
        

class PISASRPSNR(Tool):
    """One-Step Effective Diffusion Network for Real-World Image Super-Resolution [NeurIPS2024](https://arxiv.org/abs/2406.08177)"""
    def __init__(self):
        super().__init__(
            tool_name="pisasr_psnr",
            subtask="super_resolution",
            work_dir="PiSA-SR",
            script_rel_path="infer_pisasr_4kagent.py"
        )
    
    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_image", self.input_dir,
            "--output_dir", self.output_dir,
            "--pretrained_path", str(project_root / "pretrained_ckpts/PiSA-SR/pisa_sr.pkl"),
            "--pretrained_model_path", "stabilityai/stable-diffusion-2-1-base",
            "--lambda_pix", "1.0",
            "--lambda_sem", "0.0",
            "--align_method", "wavelet"
        ]


class PISASRPSNR_2x(Tool):
    """One-Step Effective Diffusion Network for Real-World Image Super-Resolution [NeurIPS2024](https://arxiv.org/abs/2406.08177)"""
    def __init__(self):
        super().__init__(
            tool_name="pisasr_2x_psnr",
            subtask="super_resolution",
            work_dir="PiSA-SR",
            script_rel_path="infer_pisasr_4kagent.py"
        )
    
    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_image", self.input_dir,
            "--output_dir", self.output_dir,
            "--upscale", "2",
            "--pretrained_path", str(project_root / "pretrained_ckpts/PiSA-SR/pisa_sr.pkl"),
            "--pretrained_model_path", "stabilityai/stable-diffusion-2-1-base",
            "--lambda_pix", "1.0",
            "--lambda_sem", "0.0",
            "--align_method", "wavelet"
        ]


class HMA(BasicSRModel):
    """[HMANet: Hybrid Multi-Axis Aggregation Network for Image Super-Resolution (CVPRW 2024)](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chu_HMANet_Hybrid_Multi-Axis_Aggregation_Network_for_Image_Super-Resolution_CVPRW_2024_paper.pdf)"""

    def __init__(self):
        super().__init__(
            tool_name="hma",
            subtask="super_resolution",
            work_dir="HMA",
            script_rel_path=Path("hma")/'infer_hma_4kagent.py'
        )
    
    def _update_pretrained_ckpt(self, cfg: dict):
        ckpt_name = cfg['path']['pretrain_network_g']
        cfg['path']['pretrain_network_g'] = str(project_root / f"pretrained_ckpts/{self.work_dir_name}/{ckpt_name}")


class HMA_2x(BasicSRModel):
    """[HMANet: Hybrid Multi-Axis Aggregation Network for Image Super-Resolution (CVPRW 2024)](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chu_HMANet_Hybrid_Multi-Axis_Aggregation_Network_for_Image_Super-Resolution_CVPRW_2024_paper.pdf)"""

    def __init__(self):
        super().__init__(
            tool_name="hma_2x",
            subtask="super_resolution",
            work_dir="HMA",
            script_rel_path=Path("hma")/'infer_hma_4kagent.py'
        )
    
    def _update_pretrained_ckpt(self, cfg: dict):
        ckpt_name = cfg['path']['pretrain_network_g']
        cfg['path']['pretrain_network_g'] = str(project_root / f"pretrained_ckpts/{self.work_dir_name}/{ckpt_name}")



subtask = 'super_resolution'
sr_toolbox = [
    DiffBIR(),
    XRestormer(subtask=subtask),
    SwinIR(subtask=subtask, pretrained_on='gan'),
    SwinIR(subtask=subtask, pretrained_on='psnr'),
    HATPSNR(),
    HATGAN(),
    OSEDiff(),
    PISASR(),
    SwinFIR(),
    NAFNet(subtask=subtask),
    HMA(),
    # DRCT(),
    # PISASRPSNR(),
]


sr_toolbox_2x = [
    DiffBIR_2x(),
    SwinIR_2x(subtask=subtask, pretrained_on='gan'),
    SwinIR_2x(subtask=subtask, pretrained_on='psnr'),
    HATPSNR_2x(),
    OSEDiff_2x(),           
    PISASR_2x(),
    SwinFIR_2x(),
    NAFNet_2x(subtask=subtask),
    HMA_2x(),
    # PISASRPSNR_2x(),
]


sr_toolbox_16x = [
    # DiffBIR_16x(),
    OSEDiff_16x(),
    PISASR_16x()
]