from pathlib import Path
import shutil
import logging
from time import localtime, strftime
import cv2
import os
import gc
import json
import random
from typing import Union, Optional
import torch
import pyiqa

from llm import GPT4, AzureGPT, DepictQA, PerceptionVLMAgent, LlamaVisionAgent
from . import prompts
from executor import executor, Tool
from utils.img_tree import ImgTree
from utils.logger import get_logger
from utils.misc import sorted_glob
from utils.custom_types import *
from utils.expert_IQA_eval import compute_iqa, compute_iqa_metric_score
from facexlib.utils.face_restoration_helper import FaceRestoreHelper


face_helper = FaceRestoreHelper(
    upscale_factor=1,
    face_size=512,
    crop_ratio=(1, 1),
    det_model='retinaface_resnet50',
    save_ext='png',
    use_parse=True,
    model_rootpath='')


def extract_face(input_path: Union[Path, str], res_path) -> None:
    in_path = str(input_path.resolve()) if isinstance(input_path, Path) else input_path
    self.face_helper.read_image(in_path)
        # get face landmarks for each face
    self.face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
        # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
        # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
        # align and warp each face
    self.face_helper.align_warp_face()
    face_list = []
    if len(self.face_helper.cropped_faces) > 0:
        for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
            save_folder = res_path / f'face_{idx:03d}'
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = save_folder / f'face.png'
            face_list.append(str(save_path))
            cv2.imwrite(str(save_path), cropped_face)
    return face_list

def face_restore(img_path: Path) -> None:
    # There is a tool list in face restoration toolbox, run all tools in the toolbox and compare the results
    # The best result is selected as the final result
    sub_task = 'face restoration'
    subtask_dir, toolbox = self._prepare_for_subtask_face_restore(sub_task)
    # Copy the input image to the subtask directory
    ori_image_path = subtask_dir / "ori"
    ori_image_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(img_path), ori_image_path / "input.png")
    # Extract face from the input image
    step_face_path = subtask_dir / "0-img"
    step_face_path.mkdir(parents=True, exist_ok=True)
    face_img_paths = self.extract_face(img_path, step_face_path)
    # print('face_img_paths: ', face_img_paths)
    # matches the self.face_list in length
    # assert len(face_img_paths) == len(self.face_list), "The number of faces extracted is not equal to the number of faces in the original image."
    if len(face_img_paths) != len(self.face_list):
        face_img_paths = face_img_paths[:len(self.face_list)]
    # Evaluate the face image
    evaluate_results = []
    # print(self.face_helper.restored_faces)
    
    for i in range(len(face_img_paths)):
        face_evaluations = []
        face_evaluation = self.evaluate_tool_result_face(face_img_paths[i], self.face_list[i])
        self.workflow_logger.info(
            f"Face: {i:03d}"
            f"{sub_task.capitalize()} result: "
            f"Origin Face "
            f"Score: {face_evaluation}.")
        # print('subtask_dir: ', subtask_dir)
        # print('face_evaluation: ', face_evaluation)
        # print('input_folder for tool: ', Path(face_img_paths[i]).parents[0])
        face_evaluations.append({'path': face_img_paths[i], 'evaluation': face_evaluation})
        for tool in toolbox:
            # prepare directory
            tool_dir = subtask_dir / f"tool-face-{i}-{tool.tool_name}"
            output_dir = tool_dir / "0-img"
            output_dir.mkdir(parents=True)
            print('output_dir: ', output_dir)
            # invoke tool
            tool(input_dir=Path(face_img_paths[i]).parents[0],output_dir=output_dir,silent=True,)
            
            # Evaluate the result (weighted score, should be a dict with {'face_00': score, 'face_01': score, ...})
            out_face_path = str(output_dir / 'output.png')
            face_evaluation = self.evaluate_tool_result_face(out_face_path, self.face_list[i])
            # self._record_tool_res(output_dir, face_evaluation)
            self.workflow_logger.info(
                f"Face: {i:03d}"
                f"{sub_task.capitalize()} result: "
                f"{tool.tool_name} "
                f"Score: {face_evaluation}.")

            face_evaluations.append({'path': out_face_path, 'evaluation': face_evaluation})
        # print('face_evaluations: ', face_evaluations)
        best_face_path = max(face_evaluations, key=lambda x: x['evaluation'])['path']
        evaluate_results.append(face_evaluations)
        self.face_helper.add_restored_face(cv2.imread(str(best_face_path)))
    
    # Merge the face images back to the original image
    # print('Input shape: ', self.face_helper.input_img.shape)
    # print('Inverse Affine: ', self.face_helper.affine_matrices)
    self.face_helper.get_inverse_affine(None)
    # paste each restored face to the input image
    restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=None)
    cv2.imwrite(str(img_path), restored_img)
    self.face_helper.clean_all()

    result_file = os.path.join(subtask_dir, "result_scores_faces.txt")
    with open(result_file, "w") as f:
        for evaluate_result in evaluate_results:
            for face_evaluation in evaluate_result:
                short_path = face_evaluation['path'].split('/')[-3]
                f.write(f"{short_path}, {face_evaluation['evaluation']}\n")
            f.write('\n')
    f.close()


def evaluate_tool_result_face(self, restored_face: Path, ori_face: str) -> float:
    # restored_face = cv2.imread(str(restored_face))
    # restored_face_tensor = torch.from_numpy(restored_face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # ori_face = cv2.imread(ori_face)
    # ori_face_tensor = torch.from_numpy(ori_face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # # compare the restored face with the original face with PSNR
    # psnr_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr')
    # psnr = psnr_metric(restored_face_tensor, ori_face_tensor)

    # Weighted sum of NIQE and Deg
    Deg_score = calculate_cos_dist(str(restored_face), ori_face)
    Niqe_score = calculate_niqe(str(restored_face))
    print('Deg_score: ', Deg_score)
    print('Niqe_score: ', Niqe_score)
    Combined_score = 20 - Niqe_score - 0.05 * Deg_score

    return Combined_score


def _prepare_for_subtask_face_restore(
    self, subtask: Subtask
) -> tuple[Path, list[Tool]]:
    self.workflow_logger.info(
        f"Executing {subtask} on {self._img_nickname(self.cur_node['img_path'])}..."
    )

    subtask_dir = Path(self.cur_node["img_path"]).parents[1] / f"subtask-{subtask}"
    subtask_dir.mkdir()

    # degradation = self.subtask_degra_dict[subtask]
    toolbox = self.executor.toolbox_router[subtask]
    random.shuffle(toolbox)

    return subtask_dir, toolbox


# Extract faces in the input image
face_dir = input_img_path.split('.')[0] + '_faces'
face_list = extract_face(input_img_path, face_dir)
face_helper.clean_all()

# Face restore and select the best 
face_restore(sr_img_path)