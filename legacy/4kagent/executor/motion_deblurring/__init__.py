import os
import shutil
from pathlib import Path

from ..tool import Tool
from ..multitask_tools import *


__all__ = ['motion_deblurring_toolbox']
project_root = Path(__file__).resolve().parents[2]


class EVSSM(Tool):
    def __init__(self):
        super().__init__(
            tool_name="EVSSM",
            subtask='motion_deblurring',
            work_dir="EVSSM",
            script_rel_path="infer_evssm_4kagent.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_dir", self.input_dir,
            "--save_image_dir", self.output_dir,
            "--ckpts", str(project_root / "pretrained_ckpts/EVSSM/net_g_GoPro.pth"),
        ]


class AdaRevD(Tool):
    def __init__(self):
        super().__init__(
            tool_name="AdaRevD",
            subtask='motion_deblurring',
            work_dir="AdaRevD",
            script_rel_path="inference.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_dir", self.input_dir,
            "--save_image_dir", self.output_dir,
            "--model_checkpoint", str(project_root / 'pretrained_ckpts/AdaRevD/RevD-L_GoPro/net_g_GoPro.pth'),
            "--state_dict_pth_classifier", str(project_root / 'pretrained_ckpts/AdaRevD/classifier/GoPro.pth'),
            "--yaml_config", str(project_root / 'pretrained_ckpts/AdaRevD/Options/AdaRevID-B-GoPro-test.yml'),
        ]


class FFTformer(Tool):
    def __init__(self):
        super().__init__(
            tool_name="FFTformer",
            subtask='motion_deblurring',
            work_dir="FFTformer",
            script_rel_path="inference.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_dir", self.input_dir,
            "--result_dir", self.output_dir,
            "--model_checkpoint", str(project_root / 'pretrained_ckpts/FFTformer/fftformer_GoPro.pth'),
        ]


class UFPDeblur(Tool):
    def __init__(self):
        super().__init__(
            tool_name="UFPDeblur",
            subtask='motion_deblurring',
            work_dir="UFPDeblur",
            script_rel_path="inference.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_dir", self.input_dir,
            "--result_dir", self.output_dir,
            "--model_checkpoint", str(project_root / 'pretrained_ckpts/UFPDeblur/train_on_GoPro/net_g_latest.pth'),
        ]


class MLWNet(Tool):
    def __init__(self):
        super().__init__(
            tool_name="MLWNet",
            subtask='motion_deblurring',
            work_dir="MLWNet",
            script_rel_path="inference.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_dir", self.input_dir,
            "--result_dir", self.output_dir,
            "--model_checkpoint", str(project_root / 'pretrained_ckpts/MLWNet/gopro-width64.pth'),
        ]


class Turtle(Tool):
    def __init__(self):
        super().__init__(
            tool_name="Turtle",
            subtask='motion_deblurring',
            work_dir="Turtle",
            script_rel_path="inference.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_dir", self.input_dir,
            "--result_dir", self.output_dir,
            "--model_checkpoint", str(project_root / 'pretrained_ckpts/Turtle/GoPro_Deblur.pth'),
            "--config", str(project_root / 'pretrained_ckpts/Turtle/Options/Turtle_Deblur_Gopro.yml')
        ]

subtask = 'motion_deblurring'
motion_deblurring_toolbox = [
    Restormer(subtask=subtask), 
    MPRNet(subtask=subtask),
    MAXIM(subtask=subtask),
    XRestormer(subtask=subtask),
    NAFNet(subtask=subtask),
    EVSSM(),
    # AdaRevD(),
    # FFTformer(),
    # UFPDeblur(),
    # MLWNet(),
    # Turtle(),
]