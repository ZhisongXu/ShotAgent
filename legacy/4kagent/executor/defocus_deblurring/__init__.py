import os
import shutil
from pathlib import Path

from ..tool import Tool
from ..multitask_tools import *


__all__ = ['defocus_deblurring_toolbox']
project_root = Path(__file__).resolve().parents[2]


class IFAN(Tool):
    """[Iterative Filter Adaptive Network for Single Image Defocus Deblurring (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Iterative_Filter_Adaptive_Network_for_Single_Image_Defocus_Deblurring_CVPR_2021_paper.pdf)"""   

    def __init__(self):
        super().__init__(
            tool_name="ifan",
            subtask='defocus_deblurring',
            work_dir="IFAN",
            script_rel_path='infer_ifan_4kagent.py'
        )

    def _preprocess(self):
        """Requires parameter `input_dir: Path`. IFAN requires that the option `input_dir` should contain a directory {data} (i.e., 'random' here) that contains the input image."""
        rqd_input_dir = self.input_dir / 'random'
        self.rqd_input_dir = rqd_input_dir
        rqd_input_dir.mkdir()
        img_name = next(f for f in os.listdir(self.input_dir) if os.path.isfile(self.input_dir / f))
        rqd_input_path = rqd_input_dir / img_name
        cur_input_path = self.input_dir / img_name
        shutil.copy(cur_input_path, rqd_input_path)

    def _get_cmd_opts(self) -> list[str]:
        """Requires parameter `input_dir: Path`, `output_dir: Path`, `opt_task: str`, and `opt_ckpt_name: str`."""        
        return [
            "--mode", "IFAN_44",
            "--network", "IFAN",
            "--config", "config_IFAN_44",
            "--data", "random",
            "--ckpt_abs_name", str(project_root / "pretrained_ckpts/IFAN/IFAN_44.pytorch"),
            "--data_offset", self.input_dir,
            "--output_offset", self.output_dir,
        ]

    def _postprocess(self):
        """Requires parameter `rqd_input_dir: Path` and `output_dir: Path`. After execution, `output_dir` will be like
        ```
        {output_dir}
        └── quanti_quali
            └── IFAN_44
                ├── random
                │   └── {YYYY_MM_DD_HHMM}
                │       ├── input
                |       |   ├── jpg
                |       |   |   └── 01.jpg
                │       |   └── png
                │       |       └── 01.png
                |       └── output
                |           ├── jpg
                |           |   └── 01.jpg
                │           └── png
                │               └── 01.png
                ├── IFAN_44.pytorch
                └── score_random.txt
        ```
        Cleans up these temporary directories.
        """

        shutil.rmtree(self.rqd_input_dir)
        random_dir = self.output_dir / 'quanti_quali' / 'IFAN_44' / 'random'
        random_dir_content = os.listdir(random_dir)
        assert len(random_dir_content) == 1, "There're more than one directory in the output directory."
        cur_output_path = random_dir / random_dir_content[0] / 'output' / 'png' / '01.png'
        output_path = self.output_dir / 'output.png'
        cur_output_path.replace(output_path)
        shutil.rmtree(self.output_dir / 'quanti_quali')


class DRBNet(Tool):
    """[Learning to Deblur using Light Field Generated and Real Defocused Images (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Ruan_Learning_to_Deblur_Using_Light_Field_Generated_and_Real_Defocus_CVPR_2022_paper.pdf)"""

    def __init__(self):
        super().__init__(
            tool_name="drbnet",
            subtask='defocus_deblurring',
            work_dir="DRBNet",
            script_rel_path='infer_drbnet_4kagent.py'
        )

    def _get_cmd_opts(self) -> list[str]:
        """Requires parameter `input_dir: Path` and `output_dir: Path`"""        
        return [
            "--eval_data", "CUHK",
            "--dataroot_cuhk", self.input_dir,
            "--results_dir", self.output_dir,
            "--ckpt_path", str(project_root / "pretrained_ckpts/DRBNet/single_image_defocus_deblurring.pth"),
            "--net_mode", "single",
            "--tile", "1024",
            "--tile_overlap", "64",
            "--save_images"
        ]
    
    def _postprocess(self):
        """Requires parameter `output_dir: Path`. After execution, `output_dir` will be like
        ```
        {output_dir}
        └── defocus_deblur
            └── CUHK
                └── single
                    └── {YYYY-MM-DD_HHMM}
                            ├── input
                            |   └── xxx
                            ├── output
                            |   └── xxx
                            └── score_CUHK.txt
        ```
        Cleans up these temporary directories.
        """

        outputs = list(self.output_dir.glob("defocus_deblur/CUHK/single/*/output/*"))
        assert len(outputs) == 1, f"There're more than one directory in the output directory"
        cur_output_path = outputs[0]
        output_path = self.output_dir / 'output.png'
        cur_output_path.replace(output_path)
        shutil.rmtree(self.output_dir / 'defocus_deblur')


class ConvIR(Tool):
    def __init__(self):
        super().__init__(
            tool_name="ConvIR",
            subtask='defocus_deblurring',
            work_dir="ConvIR",
            script_rel_path="infer_convir_4kagent.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_dir", self.input_dir,
            "--result_dir", self.output_dir,
            "--model_checkpoint", str(project_root / "pretrained_ckpts/ConvIR/pretrain_models/dpdd-large.pkl"),
        ]


subtask = 'defocus_deblurring'
defocus_deblurring_toolbox = [
    DRBNet(),
    IFAN(),
    Restormer(subtask=subtask),
    DiffPlugin(subtask=subtask),
    LaKDNet(subtask=subtask),
    # ConvIR(),
    # AutoDIR(subtask=subtask),
]