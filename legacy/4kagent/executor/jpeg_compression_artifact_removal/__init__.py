import os
from shutil import rmtree

from ..tool import Tool
from ..multitask_tools import *


__all__ = ['jpeg_compression_artifact_removal_toolbox']


class FBCNN(Tool):
    """[Towards Flexible Blind JPEG Artifacts Removal (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Towards_Flexible_Blind_JPEG_Artifacts_Removal_ICCV_2021_paper.pdf). There are seven outputs corresponding to different quality factors, one predicted and others pre-defined."""    

    def __init__(self, qf: str|int):
        """qf can be "blind", 5, or 90"""        
        super().__init__(
            tool_name=f"fbcnn_{qf}",
            subtask="jpeg_compression_artifact_removal",
            work_dir="FBCNN",
            script_rel_path="infer_fbcnn_4kagent.py"
        )
        self.qf = qf

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_dir", self.input_dir,
            "--weight_dir", 'FBCNN/model_zoo',
            "--output_dir", self.output_dir,
            "--qf", self.qf
        ]


subtask = 'jpeg_compression_artifact_removal'
jpeg_compression_artifact_removal_toolbox = [
    FBCNN(qf="blind"),
    FBCNN(qf=5),
    FBCNN(qf=90),
    SwinIR(subtask=subtask, pretrained_on='40'),
]
