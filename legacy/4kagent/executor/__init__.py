import os
from pathlib import Path
import shutil
import time
import cv2

from utils.custom_types import Subtask, ToolName

from .super_resolution import sr_toolbox, sr_toolbox_2x, sr_toolbox_16x
from .denoising import denoising_toolbox
from .motion_deblurring import motion_deblurring_toolbox
from .defocus_deblurring import defocus_deblurring_toolbox
from .dehazing import dehazing_toolbox
from .deraining import deraining_toolbox
from .brightening import brightening_toolbox
from .jpeg_compression_artifact_removal import jpeg_compression_artifact_removal_toolbox
from .face_restoration import face_restoration_toolbox
from .old_photo_restoration import old_photo_restoration_toolbox

from .tool import Tool


__all__ = ['executor']


class Executor:
    def __init__(self):
        self.toolbox_router: dict[str, list[Tool]] = {}
        self._executed_subtask_cnt: int = 0

    def register_subtask(self, subtask_name, subtask_toolbox) -> None:
        self.toolbox_router[subtask_name] = subtask_toolbox

    @property
    def subtasks(self) -> set[str]:
        return set(self.toolbox_router.keys())

    @property
    def executed_subtask_cnt(self) -> int:
        return self._executed_subtask_cnt

    def execute_subtask(self, subtask: str, input_path: Path) -> Path:
        """Invokes tools to try to execute the given subtask. `input_path` is the path to the input image, and the directory of it must be "0-img". This method will generate a directory in the same directory as "0-img", containing multiple directories, each of which contains outputs of a tool.\n
        Before:
        ```
        .
        ├── 0-img
        │   └── {input_path}
        └── ...
        ```
        After:
        ```
        .
        ├── 0-img
        │   └── {input_path}
        ├── {subtask_dir}
        |   ├── {tool_dir} 1
        |   │   └── 0-img
        |   │       └── output.png (this path could be another `input_path` parameter of this method)
        |   ├── ...
        |   └── {tool_dir} n
        |       └── 0-img
        |           └── output.png
        └── ...
        ```
        """

        self._executed_subtask_cnt += 1
        # prepare directory
        subtask_dir = input_path.parents[1] / f"subtask{self._executed_subtask_cnt}-{subtask.replace(' ', '_')}"
        subtask_dir.mkdir()

        toolbox = self.toolbox_router[subtask]
        tool_idx = 0

        for tool in toolbox:
            # prepare directory
            tool_idx += 1
            tool_dir = subtask_dir / f'tool{tool_idx}-{tool.tool_name}'
            output_dir = tool_dir / '0-img'
            output_dir.mkdir(parents=True)
            # invoke
            tool(input_dir=input_path.parent, output_dir=output_dir)
            # get output
            output_path = list(output_dir.glob('*'))[0]

        return output_path
    
    def invoke_a_tool(self, 
                      subtask_name: str, tool_name: str, 
                      input_dir: Path, output_dir: Path):
        toolbox = self.toolbox_router[subtask_name]
        for tool in toolbox:
            if tool.tool_name == tool_name:
                tool(input_dir, output_dir)
                return
            
    def test_toolbox(self, 
                     input_dir: Path, 
                     output_dir: Path, 
                     subtask: str) -> None:
        toolbox = self.toolbox_router[subtask]
        tmp_output_dir = output_dir / "tmp"
        tmp_output_dir.mkdir()
        for tool in toolbox:
            tool(input_dir, tmp_output_dir)
            outputs = list(tmp_output_dir.glob('*'))
            if len(outputs) == 1:
                outputs[0].replace(output_dir / f"{tool.tool_name}.png")
            else:
                for output in outputs:
                    output.replace(output_dir / f"{tool.tool_name}_{output.stem[7:]}.png")
        tmp_output_dir.rmdir()
            
    def test_all_tools(self, input_dir: Path, output_dir: Path):
        def _check_shape(input_shape, output_shape):
            if input_shape == output_shape:
                return True
            if input_shape[0] * 4 == output_shape[0] and input_shape[1] * 4 == output_shape[1]:
                return True
            return False
        start_time = time.time()
        imgs = sorted(list(input_dir.glob('*')))
        tool_cnt = 0
        misaligned_tools: list[tuple[Subtask, ToolName]] = []
        for i, img_path in enumerate(imgs):
            input_shape = cv2.imread(str(img_path)).shape

            subtask_name = img_path.stem
            subtask_dir = output_dir / f"{i+1}-{subtask_name.replace(' ', '_')}"
            new_input_dir = subtask_dir / '0-input'
            new_input_path = new_input_dir / 'input.png'
            new_input_dir.mkdir(parents=True)
            shutil.copy(img_path, new_input_path)
            toolbox = self.toolbox_router[subtask_name]
            tool_idx = 0

            for tool in toolbox:
                tool_cnt += 1
                tool_idx += 1
                tool_dir = subtask_dir / f'{tool_idx}_{tool.tool_name}'
                tool_dir.mkdir()
                tool(new_input_dir, output_dir=tool_dir)
                output_path = [file for file in tool_dir.glob('*') if file.suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']][0]
                output_shape = cv2.imread(str(output_path)).shape
                if not _check_shape(input_shape, output_shape):
                    misaligned_tools.append((subtask_name, tool.tool_name))
                output_path.replace(subtask_dir / f'{tool_idx}_{tool.tool_name}.png')
                tool_dir.rmdir()

            new_input_path.replace(subtask_dir / 'input.png')
            new_input_dir.rmdir()
            
        end_time = time.time()
        consumed_time = end_time - start_time
        print(f"Time elapsed: {consumed_time:.2f}s")
        print(f"Tool count: {tool_cnt}")
        print(f"Average time per tool: {consumed_time / tool_cnt:.2f}s")
        print(f"Tools that cannot keep the image size: {misaligned_tools}")


# make singleton
executor = Executor()
executor.register_subtask('super-resolution', sr_toolbox)
executor.register_subtask('super-resolution_2x', sr_toolbox_2x)
executor.register_subtask('denoising', denoising_toolbox)
executor.register_subtask('motion deblurring', motion_deblurring_toolbox)
executor.register_subtask('defocus deblurring', defocus_deblurring_toolbox)
executor.register_subtask('deraining', deraining_toolbox)
executor.register_subtask('dehazing', dehazing_toolbox)
executor.register_subtask('brightening', brightening_toolbox)
executor.register_subtask('jpeg compression artifact removal', jpeg_compression_artifact_removal_toolbox)
executor.register_subtask('face restoration', face_restoration_toolbox)
executor.register_subtask('old_photo_restoration', old_photo_restoration_toolbox)

# comparsion experiment
executor.register_subtask('super-resolution_16x', sr_toolbox_16x)
