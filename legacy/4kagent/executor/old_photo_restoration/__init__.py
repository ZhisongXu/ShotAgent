import os
import shutil

from ..tool import Tool
from ..multitask_tools import *


__all__ = ['old_photo_restoration_toolbox']


class BOBL(Tool):
    """Bringing Old Photos Back to Life [CVPR 2022] Paper: https://arxiv.org/abs/2004.09484"""
    def __init__(self):
        super().__init__(
            tool_name="bobl",
            subtask="old_photo_restoration",
            work_dir="BOBL",
            script_rel_path="infer_bobl_s1_4kagent.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_folder", self.input_dir,
            "--output_folder", self.output_dir,
            "--with_scratch"
        ]

    def _postprocess(self):
        stage1_outputs = list(self.output_dir.glob('*'))
        if not stage1_outputs:
            raise RuntimeError("No output subfolders found.")
        
        restored_image_dir = stage1_outputs[-1] / 'restored_image'

        # Handle edge case: restoration failed (empty folder), copy input as fallback
        if not any(restored_image_dir.glob('*')):
            input_images = list(self.input_dir.glob('*'))
            if input_images:
                fallback_image = input_images[0]
                fallback_target = restored_image_dir / fallback_image.name
                fallback_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(fallback_image), str(fallback_target))
            else:
                raise FileNotFoundError("No input images found to copy as fallback.")
            
        final_outputs = list(restored_image_dir.glob('*'))
        assert len(final_outputs) == 1, "Expected exactly one restored image."

        final_image = final_outputs[0]
        final_image.replace(self.output_dir / 'output.png')
        
        intermediate_dir = final_image.parent.parent
        intermediate_dir.replace(self.output_dir.parent / 'intermediate')


subtask = 'old_photo_restoration'
old_photo_restoration_toolbox = [
    BOBL(),
]
