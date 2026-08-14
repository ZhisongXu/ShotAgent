from pathlib import Path
import numpy as np
import cv2
from scipy.io import loadmat
import torch
import math
from typing import Optional
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils.matlab_functions import imresize


__all__ = [
    "lr", 
    "darken", 
    "add_noise", 
    "add_jpeg_comp_artifacts", 
    "add_haze",
    "add_motion_blur",
    "add_defocus_blur",
    "add_rain",
    # "add_raindrops"
]


def lr(img, keep_size=False):
    """
    Resize the image to 1/4 of its original size.
    If keep_size=True, resize back to original size.
    """
    img = img.copy()
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = imresize(img, scale=0.25)
    if keep_size:
        img = imresize(img, scale=4)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).round().astype(np.uint8)
    return img


def add_noise(img, noise_type: Optional[str] = None, arg=None):
    """
    Add Gaussian or Poisson noise to the image.
    noise_type: "Gaussian" or "Poisson", randomly selected if None.
    arg: sigma for Gaussian noise, or scale for Poisson noise.
    """
    img = img.copy()
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    types = ["Gaussian", "Poisson"]
    if noise_type is None:
        noise_type = np.random.choice(types)
    else:
        assert noise_type in types

    if noise_type == "Gaussian":
        sigma_range = [arg, arg] if arg is not None else [20, 50]
        out = random_add_gaussian_noise_pt(img, sigma_range=sigma_range, clip=True, rounds=False)
    else:
        scale_range = [arg, arg] if arg is not None else [1, 3]
        out = random_add_poisson_noise_pt(img, scale_range=scale_range, clip=True, rounds=False)

    lq = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    lq = (lq * 255).clip(0, 255).round().astype(np.uint8)
    return lq


def add_jpeg_comp_artifacts(img, quality_factor: Optional[int] = None):
    """
    Apply JPEG compression artifacts with quality factor in [10, 30).
    """
    img = img.copy()
    if quality_factor is None:
        quality_factor = np.random.randint(10, 30)
    _, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return img


def darken(img, darken_type: Optional[str] = None, arg=None):
    """
    Darken the image using one of three methods:
    - constant shift in brightness
    - gamma correction
    - linear mapping of V channel in HSV space
    """
    img = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    types = ["constant shift", "gamma correction", "linear mapping"]
    if darken_type is None:
        darken_type = np.random.choice(types)
    else:
        assert darken_type in types

    if darken_type == "constant shift":
        shift = arg if arg is not None else np.random.randint(30, 50)
        v = np.clip(np.int16(v) - shift, 0, 255).astype(np.uint8)
    elif darken_type == "gamma correction":
        gamma = arg if arg is not None else np.random.uniform(0.5, 0.7)
        v = (cv2.pow(v / 255.0, 1.0 / gamma) * 255).clip(0, 255).round().astype(np.uint8)
    else:  # linear mapping
        dst_max = arg if arg is not None else np.random.randint(100, 150)
        vmin, vmax = v.min(), v.max()
        v = ((v - vmin) / (vmax - vmin) * dst_max).round().astype(np.uint8)

    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def add_haze(img, idx, depth_dir=Path("dataset/depth").resolve(), A=None, beta=None):
    """
    Add haze using atmospheric scattering model:
    I(x) = J(x)*t(x) + A*(1-t(x)), where t(x) = exp(-beta * d(x))
    """
    img = img.copy()
    d = loadmat(depth_dir / idx / "predict_depth.mat")['data_obj']
    d = cv2.resize(d, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    d = d / d.max()

    A = A if A is not None else np.random.uniform(0.7, 1.0)
    beta = beta if beta is not None else np.random.uniform(0.6, 1.8)

    t = np.exp(-beta * d)[..., np.newaxis]
    hazy = img * t + A * 255 * (1 - t)
    return hazy.clip(0, 255).round().astype(np.uint8)


def add_motion_blur(img, severity: Optional[int] = None):
    """
    Add motion blur with severity in {0,1,2}.
    """
    img = img.copy()
    if severity is None:
        severity = np.random.randint(3)
    radius, sigma = [(10, 3), (15, 5), (15, 8)][severity]
    angle = np.random.uniform(-90, 90)

    width = radius * 2 + 1
    k = (np.exp(-np.arange(width) ** 2 / (2 * (sigma ** 2)))) / (np.sqrt(2 * np.pi) * sigma)  # gaussian
    kernel = k / np.sum(k)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])

    blurred = np.zeros_like(img, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil((i * point[0]) / hypot - 0.5)
        dx = -math.ceil((i * point[1]) / hypot - 0.5)
        if abs(dy) >= img.shape[0] or abs(dx) >= img.shape[1]:
            break

        # Shift image
        if dx < 0:
            shifted = np.roll(img, img.shape[1] + dx, axis=1)
            shifted[:, dx:] = shifted[:, dx - 1:dx]
        elif dx > 0:
            shifted = np.roll(img, dx, axis=1)
            shifted[:, :dx] = shifted[:, dx:dx + 1]
        else:
            shifted = img

        if dy < 0:
            shifted = np.roll(shifted, img.shape[0] + dy, axis=0)
            shifted[dy:, :] = shifted[dy - 1:dy, :]
        elif dy > 0:
            shifted = np.roll(shifted, dy, axis=0)
            shifted[:dy, :] = shifted[dy:dy + 1, :]

        blurred += kernel[i] * shifted

    img = blurred.clip(0, 255).round().astype(np.uint8)
    return img


def add_defocus_blur(img, severity: Optional[int] = None):
    """
    Add defocus blur with severity in {0,1,2}.
    """
    img = img.copy()
    if severity is None:
        severity = np.random.randint(3)
    radius, alias_blur = [(3, 0.1), (4, 0.5), (6, 0.5)][severity]

    if radius <= 8:
        L = np.arange(-8, 9)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)

    X, Y = np.meshgrid(L, L)
    aliased_disk = ((X ** 2 + Y ** 2) <= radius ** 2).astype(np.float32)
    aliased_disk /= aliased_disk.sum()

    kernel = cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

    img_norm = img / 255.0
    channels = [cv2.filter2D(img_norm[:, :, d], -1, kernel) for d in range(3)]
    channels = np.stack(channels, axis=-1)

    img_blurred = (channels.clip(0, 1) * 255).round().astype(np.uint8)
    return img_blurred


def add_rain(img, value: Optional[int] = None):
    """
    Add rain effect to the image.
    """
    img = img.copy()

    w = 3  # thickness of rain streaks
    length = np.random.randint(20, 40)
    angle = np.random.randint(-30, 30)

    value = value if value is not None else np.random.randint(50, 100)

    noise = np.random.uniform(0, 256, img.shape[:2])
    threshold = 256 - value * 0.01
    noise[noise < threshold] = 0

    # initial blur kernel
    kernel_init = np.array([[0, 0.1, 0],
                            [0.1, 8, 0.1],
                            [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, kernel_init)

    # create motion blur kernel for rain streaks
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    diag = np.diag(np.ones(length))
    kernel_rain = cv2.warpAffine(diag, trans, (length, length))
    kernel_rain = cv2.GaussianBlur(kernel_rain, (w, w), 0)

    blurred = cv2.filter2D(noise, -1, kernel_rain)

    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = blurred.astype(np.uint8)

    rain_layer = np.repeat(blurred[:, :, np.newaxis], 3, axis=2)

    img_float = img.astype(np.float32) + rain_layer
    np.clip(img_float, 0, 255, out=img_float)

    return img_float.round().astype(np.uint8)
