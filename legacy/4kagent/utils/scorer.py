import os
import cv2
import math
import torch
import pyiqa
import numpy as np

from pathlib import Path
from typing import Optional
from torch.nn import functional as F
from torch.nn import DataParallel
from torchvision.transforms.functional import normalize
from basicsr.utils.matlab_functions import imresize

from .config import Config
from .resnet import resnet_face18
from pyiqa.models.inference_model import InferenceModel


FR_METRIC_NAME_LST = ["psnr", "ssim", "lpips"]
NR_METRIC_NAME_LST = ["maniqa", "clipiqa", "musiq"]
METRIC_NAME_LST = FR_METRIC_NAME_LST + NR_METRIC_NAME_LST


class Scorer:
    """Computes image quality scores using full-reference and no-reference metrics."""

    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fr_metric_name_lst = FR_METRIC_NAME_LST
        self.nr_metric_name_lst = NR_METRIC_NAME_LST
        self.metric_name_lst = self.fr_metric_name_lst + self.nr_metric_name_lst

        self.fr_metrics = [pyiqa.create_metric(name, device=device) for name in self.fr_metric_name_lst]
        self.nr_metrics = [pyiqa.create_metric(name, device=device) for name in self.nr_metric_name_lst]
        self.metrics = self.fr_metrics + self.nr_metrics

        self.lower_better_dict = {metric.metric_name: metric.lower_better for metric in self.metrics}

    def __call__(self, img_path: Path, ref_img_path: Optional[Path] = None) -> list[tuple[str, bool, float]]:
        img = self._get_img_tensor(img_path)

        if ref_img_path:
            ref_img = self._get_img_tensor(ref_img_path)

            if img.shape != ref_img.shape:
                if img.shape[2] * 4 == ref_img.shape[2] and img.shape[3] * 4 == ref_img.shape[3]:
                    img = imresize(img[0], scale=4).unsqueeze(0).clamp(0, 1)
                else:
                    raise ValueError("Image shapes do not match.")

            metric_lst = self.metrics
        else:
            ref_img = None
            metric_lst = self.nr_metrics

        return [
            (metric.metric_name, metric.lower_better, self._get_score(metric, img, ref_img))
            for metric in metric_lst
        ]

    def _get_img_tensor(self, img_path: Path) -> torch.Tensor:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return img.unsqueeze(0)

    def _get_score(self, metric: InferenceModel, img: torch.Tensor, ref_img: Optional[torch.Tensor] = None) -> float:
        return metric(img) if metric.metric_mode == "NR" else metric(img, ref_img).item()


def img2tensor(imgs, bgr2rgb=True, float32=True):
    def _totensor(img):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img.astype("float32") if img.dtype == "float64" else img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img.float() if float32 else img

    return [_totensor(img) for img in imgs] if isinstance(imgs, list) else _totensor(imgs)


def load_image(img_path):
    image = cv2.imread(img_path, 0)
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
    image = (image.astype(np.float32) - 127.5) / 127.5
    return torch.from_numpy(image[np.newaxis, :, :])


def load_image_torch(img_path):
    image = cv2.imread(img_path) / 255.0
    image = img2tensor(image, bgr2rgb=True, float32=True)
    normalize(image, [0.5]*3, [0.5]*3, inplace=True)
    image = image.unsqueeze(0)
    image = 0.2989 * image[:, 0] + 0.5870 * image[:, 1] + 0.1140 * image[:, 2]
    image = image.unsqueeze(1)
    return F.interpolate(image, (128, 128), mode='bilinear', align_corners=False)


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def calculate_cos_dist(restored_path, gt_path, model_path="../pretrained_ckpts/Face_eval/resnet18_110.pth"):
    if not os.path.exists(model_path):
        script_dir = os.path.abspath(os.path.dirname(__file__))
        parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
        alt_model_path = os.path.join(parent_dir, "pretrained_ckpts", "Face_eval", "resnet18_110.pth")

        if os.path.exists(alt_model_path):
            print(f"[Info] model_path not found, fallback to: {alt_model_path}")
            model_path = alt_model_path
        else:
            raise FileNotFoundError(f"Model file not found at {model_path} or fallback path {alt_model_path}")

    opt = Config()
    model = resnet_face18(opt.use_se)
    model = DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device("cuda")).eval()

    img1 = load_image(gt_path)
    img2 = load_image(restored_path)
    data = torch.stack([img1, img2]).to(torch.device("cuda"))
    
    output = model(data).cpu().detach().numpy()
    dist = np.arccos(cosin_metric(output[0], output[1])) / math.pi * 180
    return dist


def img2tensorniqe(img, bgr2rgb=True, float32=True):
    if img.shape[2] == 3 and bgr2rgb:
        img = cv2.cvtColor(img.astype("float32") if img.dtype == "float64" else img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose(2, 0, 1)).float() if float32 else img
    return img / 255.0


def calculate_niqe(restored_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iqa_niqe = pyiqa.create_metric("niqe").to(device)

    img = cv2.imread(restored_path)
    img_tensor = img2tensorniqe(img, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
    niqe = iqa_niqe(img_tensor).item()
    print(f"{restored_path} - NIQE: {niqe:.4f}")
    return niqe


if __name__ == "__main__":
    scorer = Scorer()
    
    restored_path = "/path/to/restored_image.png"
    gt_path = "/path/to/ground_truth.png"
    model_path = "/path/to/resnet18_110.pth"
    
    print(scorer(Path(restored_path), Path(gt_path)))
    print("Cosine Distance:", calculate_cos_dist(restored_path, gt_path, model_path))
    print("NIQE Score:", calculate_niqe(restored_path))
