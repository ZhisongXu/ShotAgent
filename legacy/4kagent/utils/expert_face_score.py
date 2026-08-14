import os
import sys
import shutil
import inspect
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import Compose, ToTensor, Normalize
from huggingface_hub import hf_hub_download
from transformers import AutoModel
from .clib_fiqa.face_iqa_analyzer import CLIBFIQAScorer


def download_huggingface_repo(repo_id, local_path, hf_token=None):
    """
    Downloads required files from HuggingFace Hub to the local directory.
    """
    os.makedirs(local_path, exist_ok=True)
    files_txt = os.path.join(local_path, 'files.txt')
    
    if not os.path.exists(files_txt):
        hf_hub_download(repo_id, 'files.txt', token=hf_token, local_dir=local_path, local_dir_use_symlinks=False)
    
    with open(files_txt, 'r') as f:
        file_list = f.read().splitlines()

    files_to_download = [f for f in file_list if f] + ['config.json', 'wrapper.py', 'model.safetensors']
    
    for filename in files_to_download:
        full_path = os.path.join(local_path, filename)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, filename, token=hf_token, local_dir=local_path, local_dir_use_symlinks=False)


def load_model_from_local_path(path, hf_token=None):
    """
    Loads a model using HuggingFace Transformers with trust_remote_code enabled.
    """
    print(f"Loading model from: {path}")
    model = AutoModel.from_pretrained(path, trust_remote_code=True, token=hf_token)
    return model


def load_model_by_repo_id(repo_id, save_path, hf_token=None, force_download=False):
    """
    Wrapper that optionally re-downloads repo and loads model from local path.
    """
    if force_download and os.path.exists(save_path):
        shutil.rmtree(save_path)
    download_huggingface_repo(repo_id, save_path, hf_token)
    return load_model_from_local_path(save_path, hf_token)


def pil_to_tensor_input(pil_image):
    """
    Converts a PIL image to a normalized tensor input.
    """
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(pil_image).unsqueeze(0)


def compute_identity_score(input1, input2, aligner, fr_model, return_bbox=False):
    """
    Computes cosine similarity between two face embeddings.
    """
    aligned_x1, _, aligned_ldmks1, _, _, bbox1 = aligner(input1)
    aligned_x2, _, aligned_ldmks2, _, _, bbox2 = aligner(input2)

    # Check if model expects keypoints
    sig = inspect.signature(fr_model.model.net.forward)
    if 'keypoints' in sig.parameters:
        feat1 = fr_model(aligned_x1, aligned_ldmks1)
        feat2 = fr_model(aligned_x2, aligned_ldmks2)
    else:
        feat1 = fr_model(aligned_x1)
        feat2 = fr_model(aligned_x2)

    similarity = torch.nn.functional.cosine_similarity(feat1, feat2)

    if return_bbox:
        return similarity, bbox1, bbox2
    return similarity


def compute_face_scores(img_path, ckpt_dir="../pretrained_ckpts/Face_eval/clib_fiqa"):
    """
    Runs face image quality assessment using CLIB-FIQA.
    """
    clip_model_path = os.path.join(ckpt_dir, "RN50.pt")
    clip_weights_path = os.path.join(ckpt_dir, "CLIB-FIQA_R50.pth")

    if not os.path.exists(clip_model_path) or not os.path.exists(clip_weights_path):
        script_dir = os.path.abspath(os.path.dirname(__file__))
        parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
        alt_ckpt_dir = os.path.join(parent_dir, "pretrained_ckpts", "Face_eval", "clib_fiqa")
        alt_clip_model_path = os.path.join(alt_ckpt_dir, "RN50.pt")
        alt_clip_weights_path = os.path.join(alt_ckpt_dir, "CLIB-FIQA_R50.pth")

        if os.path.exists(alt_clip_model_path) and os.path.exists(alt_clip_weights_path):
            print(f"[Info] ckpt_dir not found, fallback to: {alt_ckpt_dir}")
            clip_model_path = alt_clip_model_path
            clip_weights_path = alt_clip_weights_path
        else:
            raise FileNotFoundError(
                f"Model files not found in {ckpt_dir} or fallback path {alt_ckpt_dir}"
            )

    scorer = CLIBFIQAScorer(
        clip_model_path=clip_model_path,
        clip_weights_path=clip_weights_path,
        device="cuda"
    )
    return scorer.analyze(img_path)


if __name__ == "__main__":
    image_path = "output.png"
    result = compute_face_scores(image_path)
    print(result)
