import os
import shutil
from pathlib import Path

from ..tool import Tool
from ..multitask_tools import *


__all__ = ['dehazing_toolbox']
project_root = Path(__file__).resolve().parents[2]


class DehazeFormer(Tool):
    """[Vision Transformers for Single Image Dehazing (TIP 2023)](https://doi.org/10.1109/TIP.2023.3256763)"""    

    def __init__(self):
        super().__init__(
            tool_name="dehazeformer",
            subtask="dehazing",
            work_dir="DehazeFormer",
            script_rel_path="inference.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--data_dir", self.input_dir,
            "--result_dir", self.output_dir,
            "--save_dir",  str(project_root / "pretrained_ckpts/DehazeFormer"),
            "--tile_size", "1024",
            "--tile_overlap", "64"
        ]
    

class RIDCP(Tool):
    """[RIDCP: Revitalizing Real Image Dehazing via High-Quality Codebook Priors (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_RIDCP_Revitalizing_Real_Image_Dehazing_via_High-Quality_Codebook_Priors_CVPR_2023_paper.pdf)"""    

    def __init__(self):
        super().__init__(
            tool_name="ridcp",
            subtask="dehazing",
            work_dir="RIDCP_dehazing",
            script_rel_path="infer_ridcp_4kagent.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "-i", self.input_dir,
            "-o", self.output_dir,
            "--weight", str(project_root / "pretrained_ckpts/RIDCP_dehazing/pretrained_RIDCP.pth"),
            "--matching_weight_path", str(project_root / "pretrained_ckpts/RIDCP_dehazing/weight_for_matching_dehazing_Flickr.pth"),
            "--use_weight",
            "--alpha", "-21.25"
        ]
    
    
class MWFormer(Tool):
    def __init__(self, subtask: str):
        super().__init__(
            tool_name="MWFormer",
            subtask=subtask,
            work_dir="MWFormer",
            script_rel_path="infer_mwformer_4kagent.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--val_data_dir", self.input_dir,
            "--result_dir", self.output_dir,
            "--restore-from-stylefilter", str(project_root / "MWFormer/MWFormer_L/style_filter"),
            "--restore-from-backbone", str(project_root / "MWFormer/MWFormer_L/backbone"),
        ]
        

subtask = 'dehazing'
dehazing_toolbox = [
    XRestormer(subtask=subtask),
    RIDCP(),
    DehazeFormer(),
    MAXIM(subtask=subtask),
    DiffPlugin(subtask=subtask),
    # AutoDIR(subtask=subtask),
    # MWFormer(subtask=subtask)
]