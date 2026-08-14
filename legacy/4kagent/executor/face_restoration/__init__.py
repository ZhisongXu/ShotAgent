import os
import shutil
from pathlib import Path

from ..tool import Tool
from ..multitask_tools import *


__all__ = ['face_restoration_toolbox']
project_root = Path(__file__).resolve().parents[2]


class CodeFormer(Tool):
    """Towards Robust Blind Face Restoration with Codebook Lookup Transformer [NeurIPS 2022](https://arxiv.org/abs/2206.11253)"""
    def __init__(self):
        super().__init__(
            tool_name="codeformer",
            subtask="face_restoration",
            work_dir="CodeFormer",
            script_rel_path="infer_codeformer_4kagent.py"
        )
    
    def _get_cmd_opts(self) -> list[str]:
        return [
            "-i", self.input_dir,
            "-o", self.output_dir,
            "-w", "0.5",
            "-s", "1",
            "--has_aligned",
            # "--model_dir", "pretrained_ckpts/CodeFormer"
        ]
        

class GFPGAN(Tool):
    """GFPGAN: High-Performance Face Restoration with Global and Local Perception [CVPR 2021](https://arxiv.org/abs/2105.05233)"""
    def __init__(self):
        super().__init__(
            tool_name="gfpgan",
            subtask="face_restoration",
            work_dir="GFPGAN",
            script_rel_path="infer_gfpgan_4kagent.py"
        )
    
    def _get_cmd_opts(self) -> list[str]:
        return [
            "-i", self.input_dir,
            "-o", self.output_dir,
            "-v", "1.3",
            "-s", "1",
            "--aligned"
        ]
    
    def _postprocess(self):
        """GFPGAN will output the image into {output_dir}/{restored_faces}."""
        cur_outputs = list(self.output_dir.glob('*/*'))
        assert len(cur_outputs) == 1
        cur_output_path = cur_outputs[0]
        cur_output_path.replace(self.output_dir / 'output.png')
        cur_output_path.parent.rmdir()


class DifFace(Tool):
    """DIF-Face: De-Identification and Face Restoration for Visual Privacy Protection [CVPR 2021](https://arxiv.org/abs/2105.05233)"""
    def __init__(self):
        super().__init__(
            tool_name="difface",
            subtask="face_restoration",
            work_dir="DifFace",
            script_rel_path="infer_difface_4kagent.py"
        )
    
    def _get_cmd_opts(self) -> list[str]:
        return [
            "-i", self.input_dir,
            "-o", self.output_dir,
            "--eta", "0.5",
            "--aligned",
            "--use_fp16",
            "--task", "restoration",
            "--cfg_path", "configs/sample/iddpm_ffhq512_swinir_4kagent.yaml",
            # "--model_dir", str(project_root / "pretrained_ckpts/DifFace")
        ]
    
    def _postprocess(self):
        """DifFace will output the image into {output_dir}/{restored_faces}."""
        cur_outputs = list(self.output_dir.glob('*/*'))
        assert len(cur_outputs) == 1
        cur_output_path = cur_outputs[0]
        cur_output_path.replace(self.output_dir / 'output.png')
        cur_output_path.parent.rmdir()


subtask = 'face_restoration'
face_restoration_toolbox = [
    CodeFormer(),
    GFPGAN(),
    DifFace(),
]