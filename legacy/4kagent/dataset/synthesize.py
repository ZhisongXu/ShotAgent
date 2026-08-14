import os
import cv2
from tqdm import tqdm

from .image_degradations import *


def degrade(img, degradation, idx):
    router = {
        "low resolution": lr,
        "dark": darken,
        "noise": add_noise,
        "jpeg compression artifact": add_jpeg_comp_artifacts,
        "haze": add_haze,
        "motion blur": add_motion_blur,
        "defocus blur": add_defocus_blur,
        "rain": add_rain,
    }
    if degradation == "haze":
        return add_haze(img, idx=idx)
    return router[degradation](img)


base_dir = "dataset"
hq_dir = os.path.join(base_dir, "HQ")
degras_path = os.path.join(base_dir, "degradations.txt")
lq_dir = os.path.join(base_dir, "LQ")

os.makedirs(lq_dir, exist_ok=True)

with open(degras_path, 'r') as f:
    lines = f.readlines()

combs = []
for line in lines:
    items = [i.strip() for i in line.strip().split('+')]
    degras = [i for i in items if i]
    if degras:
        combs.append(degras)

hq_files = sorted(os.listdir(hq_dir))

for comb in combs:
    n_degra = len(comb)
    comb_dir = os.path.join(lq_dir, f"d{n_degra}", "+".join(comb))
    os.makedirs(comb_dir, exist_ok=True)
    desc = " + ".join(comb)

    for hq_file in tqdm(hq_files, desc=desc, unit='img'):
        hq_path = os.path.join(hq_dir, hq_file)
        img = cv2.imread(hq_path)
        filename_stem, _ = os.path.splitext(hq_file)
        for degra in comb:
            img = degrade(img, degra, idx=filename_stem)
        save_path = os.path.join(comb_dir, hq_file)
        cv2.imwrite(save_path, img)
