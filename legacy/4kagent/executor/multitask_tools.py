import os
import shutil
from pathlib import Path

import yaml
from typing import Union, Optional

from .tool import Tool


project_root = Path(__file__).resolve().parents[1]


class BasicSRModel(Tool):
    """Model based on [BasicSR template](https://github.com/XPixelGroup/BasicSR). 
    
    Note that a file `{work_dir}/{tool_name}/inference.py` modified from `{work_dir}/{tool_name}/test.py` is added to allow customizing output directory during inference.
    """

    def __init__(self,
                 tool_name: str,
                 subtask: str,
                 work_dir: str,
                 script_rel_path: Optional[str] = None):
        super().__init__(
            tool_name=tool_name,
            subtask=subtask,
            work_dir=work_dir,
            script_rel_path=Path(tool_name)/'inference.py' if script_rel_path is None else script_rel_path
        )
        self.work_dir_name = work_dir

    def _preprocess(self):
        """BasicSR requires a configuration file."""
        
        # build the configuration file
        cfg_path = Path().resolve() / 'executor' / self.subtask / 'configs' / f'{self.tool_name}.yml'
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # update paths
        cfg['datasets']['test_1']['dataroot_lq'] = self.input_dir
        cfg['path']['results'] = str(self.output_dir)

        # hook for subclasses to update checkpoint path
        self._update_pretrained_ckpt(cfg)

        # write updated config
        self.new_cfg_dir: Path = self.output_dir / "cfg"
        self.new_cfg_dir.mkdir()
        self.new_cfg_path: Path = self.new_cfg_dir / "cfg.yml"
        with open(self.new_cfg_path, 'w') as f:
            yaml.dump(cfg, f)

    def _update_pretrained_ckpt(self, cfg: dict):
        """Subclasses may override this to modify ckpt paths like `pretrain_network_g`."""
        pass

    def _get_cmd_opts(self) -> list[str]:
        """Requires parameter `new_cfg_path: Path`."""
        return [
            "-opt", self.new_cfg_path
        ]

    def _postprocess(self):
        """Move output image and clean up temporary config/output dirs."""

        shutil.rmtree(self.new_cfg_dir)

        cur_outputs = list(self.output_dir.rglob('*.png'))
        assert len(cur_outputs) == 1, \
            f'There should be only the output image in {self.output_dir}'
        cur_output_path = cur_outputs[0]
        shutil.move(cur_output_path, self.output_dir / 'output.png')

        # clean up any BasicSR result subdir
        for p in self.output_dir.glob("*"):
            if p.is_dir():
                shutil.rmtree(p)


class XRestormer(BasicSRModel):
    """[A Comparative Study of Image Restoration Networks for General Backbone Network Design (ECCV 2024)](https://arxiv.org/abs/2310.11881) for SR, denoising, dehazing, motion deblurring, deraining.

    Args:
        subtask (str): Subtask that can be handled by X-Restormer, one of `super_resolution`, `denoising`, `dehazing`, `motion_deblurring`, and `deraining`.
    """

    def __init__(self, subtask: str):
        super().__init__(
            tool_name="xrestormer",
            subtask=subtask,
            work_dir="X-Restormer"
        )
    
    def _update_pretrained_ckpt(self, cfg: dict):
        ckpt_name = cfg['path']['pretrain_network_g']
        cfg['path']['pretrain_network_g'] = str(project_root / f"pretrained_ckpts/{self.work_dir_name}/{ckpt_name}")


class NAFNet(BasicSRModel):
    """[NAFNet: Nonlinear Activation Free Network for Image Restoration (ECCV 2022)](https://arxiv.org/abs/2204.04676)"""

    def __init__(self, subtask: str):
        super().__init__(
            tool_name="nafnet",
            subtask=subtask,
            work_dir="NAFNet",
        )
    
    def _update_pretrained_ckpt(self, cfg: dict):
        ckpt_name = cfg['path']['pretrain_network_g']
        cfg['path']['pretrain_network_g'] = str(project_root / f"pretrained_ckpts/{self.work_dir_name}/{ckpt_name}")


class NAFNet_2x(BasicSRModel):
    """[NAFNet: Nonlinear Activation Free Network for Image Restoration (ECCV 2022)](https://arxiv.org/abs/2204.04676)"""

    def __init__(self, subtask: str):
        super().__init__(
            tool_name="nafnet_2x",
            subtask=subtask,
            work_dir="NAFNet",
            script_rel_path=Path("nafnet")/'inference.py'
        )
    
    def _update_pretrained_ckpt(self, cfg: dict):
        ckpt_name = cfg['path']['pretrain_network_g']
        cfg['path']['pretrain_network_g'] = str(project_root / f"pretrained_ckpts/{self.work_dir_name}/{ckpt_name}")


class SwinIR(Tool):
    """[SwinIR: Image Restoration Using Swin Transformer (ICCVW 2021)](https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Liang_SwinIR_Image_Restoration_Using_Swin_Transformer_ICCVW_2021_paper.pdf) for real SR, denoising, and JPEG compression artifact removal. For each task, there are multiple outputs corresponding to different pretrained models. Note that a file `inference.py` modified from `main_test_swinir.py` is added in the original directory `SwinIR` for inference.

    Args:
        subtask (str): Subtask that can be handled by MAXIM, one of `super_resolution`, `denoising`, and `jpeg_compression_artifact_removal`.
        pretrained_on (str): 'gan', 'psnr' if `subtask` is `super_resolution`; '15', '50' if `subtask` is `denoising`; '40' if `subtask` is `jpeg_compression_artifact_removal`.
    """

    def __init__(self, subtask: str, pretrained_on: str):
        super().__init__(
            tool_name=f"swinir_{pretrained_on}",
            subtask=subtask,
            work_dir="SwinIR",
            script_rel_path="infer_swinir_4kagent.py",
        )
        # corresponding `task` and `model_path` option for the subtask
        opt_dict = {
            'super_resolution':
            ('real_sr', {
                'gan': '003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth',
                'psnr': '003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth'
            }),
            'denoising':
            ('color_dn', {
                '15': '005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth',
                '50': '005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth'
            }),
            'jpeg_compression_artifact_removal':
            ('color_jpeg_car', {
                '40': '006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth'
            }),
        }
        self.opt_task, opt_model_names = opt_dict[subtask]
        self.model_name = opt_model_names[pretrained_on]

    def _get_cmd_opts(self) -> list[str]:
        """Requires parameter `input_dir: Path`, `output_dir: Path`, `opt_task: str`, and `model_name: str`."""
        opts = [
            "--task", self.opt_task,
            "--model_path", f"SwinIR/model_zoo/swinir/{self.model_name}",
            "--folder_lq", self.input_dir,
            "--save_dir", self.output_dir
        ]
        if self.subtask == "super_resolution":
            opts += [
                '--scale', '4',
                '--tile', '512',
                '--tile_overlap', '32',
                '--large_model'
            ]
        elif self.subtask == "jpeg_compression_artifact_removal":
            opts += [
                '--tile', '896',
                '--tile_overlap', '56'
            ]
        else:
            opts += [
                '--tile', '1024',
                '--tile_overlap', '64'
            ]
        return opts


class SwinIR_2x(Tool):
    """[SwinIR: Image Restoration Using Swin Transformer (ICCVW 2021)](https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Liang_SwinIR_Image_Restoration_Using_Swin_Transformer_ICCVW_2021_paper.pdf) for real SR, denoising, and JPEG compression artifact removal. For each task, there are multiple outputs corresponding to different pretrained models. Note that a file `inference.py` modified from `main_test_swinir.py` is added in the original directory `SwinIR` for inference.

    Args:
        subtask (str): Subtask that can be handled by MAXIM, one of `super_resolution`, `denoising`, and `jpeg_compression_artifact_removal`.
        pretrained_on (str): 'gan', 'psnr' if `subtask` is `super_resolution`; '15', '50' if `subtask` is `denoising`; '40' if `subtask` is `jpeg_compression_artifact_removal`.
    """

    def __init__(self, subtask: str, pretrained_on: str):
        super().__init__(
            tool_name=f"swinir_2x_{pretrained_on}",
            subtask=subtask,
            work_dir="SwinIR",
            script_rel_path="infer_swinir_4kagent.py",
        )
        # corresponding `task` and `model_path` option for the subtask
        opt_dict = {
            'super_resolution': 
            ('real_sr', {
                'gan': '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth',
                'psnr': '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_PSNR.pth'
            }),
        }
        self.opt_task, opt_model_names = opt_dict[subtask]
        self.model_name = opt_model_names[pretrained_on]

    def _get_cmd_opts(self) -> list[str]:
        """Requires parameter `input_dir: Path`, `output_dir: Path`, `opt_task: str`, and `model_name: str`."""
        opts = [
            "--task", self.opt_task,
            "--model_path", f"SwinIR/model_zoo/swinir/{self.model_name}",
            "--folder_lq", self.input_dir,
            "--save_dir", self.output_dir
        ]
        if self.subtask == "super_resolution":
            opts += [
                '--scale', '2',
                '--tile', '512',
                '--tile_overlap', '32'
            ]
        else:
            opts += [
                '--tile', '1024',
                '--tile_overlap', '64'
            ]
        return opts


class Restormer(Tool):
    """[Restormer: Efficient Transformer for High-Resolution Image Restoration (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf) for denoising, motion deblurring, defocus deblurring, and deraining.

    Args:
        subtask (str): Subtask that can be handled by Restormer, one of `denoising`, `motion_deblurring`, `defocus_deblurring`, and `deraining`.
    """

    def __init__(self, subtask: str):
        super().__init__(
            tool_name='restormer',
            subtask=subtask,
            work_dir='Restormer',
            script_rel_path='infer_restormer_4kagent.py'
        )
        # corresponding `task` option for the subtask
        opt_dict = {
            'denoising': 'Real_Denoising',
            'motion_deblurring': 'Motion_Deblurring',
            'defocus_deblurring': 'Single_Image_Defocus_Deblurring',
            'deraining': 'Deraining'
        }
        self.opt_task = opt_dict[subtask]

    def _get_cmd_opts(self) -> list[str]:
        """Requires parameter `input_dir: Path`, `output_dir: Path`, and `opt_task: str`."""
        return [
            "--task", self.opt_task,
            "--input_dir", self.input_dir,
            "--result_dir", self.output_dir,
            "--tile", "1024",
            "--tile_overlap", "64",
            "--ckpt", str(project_root / f"pretrained_ckpts/Restormer/{self.opt_task.lower()}.pth"),
        ]

    def _postprocess(self):
        """Restormer will output the image into {output_dir}/{task}."""
        cur_outputs = list(self.output_dir.glob('*/*'))
        assert len(cur_outputs) == 1
        cur_output_path = cur_outputs[0]
        cur_output_path.replace(self.output_dir / 'output.png')
        cur_output_path.parent.rmdir()


class MPRNet(Tool):
    """[Multi-Stage Progressive Image Restoration (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Zamir_Multi-Stage_Progressive_Image_Restoration_CVPR_2021_paper.pdf) for denoising, motion deblurring, and deraining.

    Args:
        subtask (str): Subtask that can be handled by MPRNet, one of `denoising`, `motion_deblurring`, and `deraining`.
    """

    def __init__(self, subtask: str):
        super().__init__(
            tool_name='mprnet',
            subtask=subtask,
            work_dir='MPRNet',
            script_rel_path='infer_mprnet_4kagent.py'
        )
        # corresponding `task` option for the subtask
        opt_dict = {
            'denoising': 'Denoising',
            'motion_deblurring': 'Deblurring',
            'deraining': 'Deraining'
        }
        self.opt_task = opt_dict[subtask]

    def _get_cmd_opts(self) -> list[str]:
        """Requires parameter `input_dir: Path`, `output_dir: Path`, and `opt_task: str`."""
        return [
            "--task", self.opt_task,
            "--input_dir", self.input_dir,
            "--result_dir", self.output_dir,
            "--model_arch_file", str(project_root / f"executor/denoising/tools/MPRNet/{self.opt_task}/MPRNet.py"),
            "--ckpt", str(project_root / f"pretrained_ckpts/MPRNet/model_{self.opt_task.lower()}.pth"),
            "--tile", "1024",
            "--tile_overlap", "64"
        ]


class MAXIM(Tool):
    """[MAXIM: Multi-Axis MLP for Image Processing (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Tu_MAXIM_Multi-Axis_MLP_for_Image_Processing_CVPR_2022_paper.pdf) for denoising, motion deblurring, deraining, raindrop removal, dehazing, low light enhancement, and image retouching.

    Args:
        subtask (str): Subtask that can be handled by MAXIM, one of `denoising`, `motion_deblurring`, `deraining`, and `dehazing`.
    """

    def __init__(self, subtask: str):
        super().__init__(
            tool_name="maxim",
            subtask=subtask,
            work_dir="maxim",
            script_rel_path=Path('maxim')/'infer_maxim_4kagent.py'
        )
        # corresponding `task` and `ckpt_path` option for the subtask
        opt_dict = {
            'brightening': ('Enhancement', 'maxim_ckpt_Enhancement_LOL_checkpoint.npz'),
            'dehazing': ('Dehazing', 'maxim_ckpt_Dehazing_SOTS-Outdoor_checkpoint.npz'),
            'denoising': ('Denoising', 'maxim_ckpt_Denoising_SIDD_checkpoint.npz'),
            'motion_deblurring': ('Deblurring', 'maxim_ckpt_Deblurring_RealBlur_R_checkpoint.npz'),
            'deraining': ('Deraining', 'maxim_ckpt_Deraining_Rain13k_checkpoint.npz'),
        }
        self.opt_task, self.opt_ckpt_name = opt_dict[subtask]

    def _preprocess(self):
        """Requires parameter `input_dir: Path`. MAXIM requires that the option `input_dir` should contain a directory `input` that contains the input image."""
        img_name = os.listdir(self.input_dir)[0]
        rqd_input_dir = self.input_dir / 'input'
        self.rqd_input_dir = rqd_input_dir
        rqd_input_dir.mkdir()
        rqd_input_path = rqd_input_dir / img_name
        cur_input_path = self.input_dir / img_name
        shutil.copy(cur_input_path, rqd_input_path)

    def _get_cmd_opts(self) -> list[str]:
        """Requires parameter `input_dir: Path`, `output_dir: Path`, `opt_task: str`, and `opt_ckpt_name: str`."""
        return [
            "--task", self.opt_task,
            "--ckpt_path", str(project_root / f"pretrained_ckpts/MAXIM/{self.opt_ckpt_name}"),
            "--input_dir", self.input_dir,
            "--output_dir", self.output_dir,
            "--has_target=False"
        ]

    def _postprocess(self):
        """Requires parameter `rqd_input_dir: Path`. Cleans up the temporary input directory `rqd_input_dir`."""
        shutil.rmtree(self.rqd_input_dir)


class AutoDIR(Tool):
    def __init__(self, subtask):
        super().__init__(
            tool_name="autodir",
            subtask=subtask,
            work_dir="AutoDIR",
            script_rel_path="infer_autodir_4kagent.py"
        )
        self.subtask = subtask

    def _get_cmd_opts(self) -> list[str]:
        customize_map = {
            "brightening": "A photo needs underexposure reduction",
            "defocus_deblurring": "A photo needs blur artifact reduction",
            "dehazing": "A photo needs haze reduction",
            "denoising": "A photo needs nosie reduction",
            "deraining": "A photo needs rain reduction",
        }
        customize = customize_map[self.subtask]
        
        return [
            "--customize", customize,
            "--input_dir", self.input_dir,
            "--steps", 100,
            "--output", self.output_dir,
            "--ckpt", str(project_root / "pretrained_ckpts/AutoDIR/autodir.ckpt"),
            "--cfg-text", 1,
            "--config", str(project_root / "executor/defocus_deblurring/tools/AutoDIR/configs/generate.yaml"),
            "--need-resize"
        ]


class DiffPlugin(Tool):
    def __init__(self, subtask: str):
        super().__init__(
            tool_name="diffplugin",
            subtask=subtask,
            work_dir="Diff-Plugin",
            script_rel_path="infer_diffplugin_4kagent.py"
        )
        self.subtask = subtask

    def _get_cmd_opts(self) -> list[str]:
        ckpt_dir = None
        if self.subtask == "defocus_deblurring":
            ckpt_dir = str(project_root / "pretrained_ckpts/Diff-Plugin/deblur")
        elif self.subtask == "deraining":
            ckpt_dir = str(project_root / "pretrained_ckpts/Diff-Plugin/derain")
        elif self.subtask == "dehazing":
            ckpt_dir = str(project_root / "pretrained_ckpts/Diff-Plugin/dehaze")
        elif self.subtask == "brightening":
            ckpt_dir = str(project_root / "pretrained_ckpts/Diff-Plugin/lowlight")

        return [
            "--pretrained_model_name_or_path", "CompVis/stable-diffusion-v1-4",
            "--clip_path", "openai/clip-vit-large-patch14",
            "--num_inference_steps", "20",
            "--img_path_dir", self.input_dir,
            "--save_root", self.output_dir,
            "--ckpt_dir", ckpt_dir,
        ]


class LaKDNet(Tool):
    def __init__(self, subtask):
        super().__init__(
            tool_name="lakdnet",
            subtask=subtask,
            work_dir="LaKDNet",
            script_rel_path="infer_lakdnet_4kagent.py"
        )
        self.subtask = subtask

    def _get_cmd_opts(self) -> list[str]:
        if self.subtask == "defocus_deblurring":
            task_type = "Defocus"
            train_dataset = "dpdd"
            ckpt = str(project_root / "pretrained_ckpts/LaKDNet/train_on_dpdd_l.pth")
        
        if self.subtask == "motion_deblurring":
            task_type = "Motion"
            train_dataset = "realr"
            ckpt = str(project_root / "pretrained_ckpts/LaKDNet/train_on_realr_l.pth")
            # ckpt = str(project_root / 'pretrained_ckpts/LaKDNet/train_on_gopro_l.pth')
        
        return [
            "--task_type", task_type,
            "--train_dataset", train_dataset,
            "--input_dir", self.input_dir,
            "--output_dir", self.output_dir,
            "--ckpt", ckpt,
        ]