import torch
import pyiqa
import gc
from PIL import Image
import torchvision.transforms as transforms

# Define available IQA metrics
AVAILABLE_METRICS = {
    # "Q-align": "qalign",
    "CLIPIQA+": "clipiqa+",
    "TOPIQ_NR": "topiq_nr",
    "MUSIQ": "musiq",
    "NIQE": "niqe"
}

TARGET_METRICS = {
    # "Q-align": "qalign",
    "CLIPIQA+": "clipiqa+",
    "MANIQA": "maniqa",
    "MUSIQ": "musiq",
    "NIQE": "niqe"
}

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

def image_to_tensor(image_path):
    """Convert an image to a PyTorch tensor."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

def compute_iqa(image_path):
    """Compute IQA metrics for the given image."""
    image_tensor = image_to_tensor(image_path)
    results = {}
    image_height = image_tensor.size(2)
    image_width = image_tensor.size(3)

    for metric_name, metric_key in AVAILABLE_METRICS.items():
        iqa_model = pyiqa.create_metric(metric_key, device=device)
        iqa_model.to(device)
        _, _, h, w = image_tensor.size()

        if max(h, w) <= 120:
            # upscale the image_tensor with scale factor 4, bicubic
            image_tensor = torch.nn.functional.interpolate(image_tensor, scale_factor=4, mode='bicubic', align_corners=False)
            image_tensor = image_tensor.clamp(0, 1)

        if max(h, w) <= 240:
            # upscale the image_tensor with scale factor 2, bicubic
            image_tensor = torch.nn.functional.interpolate(image_tensor, scale_factor=2, mode='bicubic', align_corners=False)
            image_tensor = image_tensor.clamp(0, 1)
        
        if max(h, w) > 4200:
            # downscale the image_tensor with scale factor 0.5, bicubic
            image_tensor = torch.nn.functional.interpolate(image_tensor, scale_factor=0.5, mode='bicubic', align_corners=False)
            image_tensor = image_tensor.clamp(0, 1)
        
        score = iqa_model(image_tensor).item()
        results[metric_name] = round(score, 4)

    # Release memory
    del image_tensor, iqa_model
    torch.cuda.empty_cache()

    # Format results
    result_str = "\n".join([f"{metric}: {score}" for metric, score in results.items()])
    return result_str, image_height, image_width


def compute_iqa_metric_score(image_path):
    """Compute IQA metrics for the given image. Metric target"""
    image_tensor = image_to_tensor(image_path)
    results = {}
    # image_height = image_tensor.size(2)
    # image_width = image_tensor.size(3)
    _, _, h, w = image_tensor.size()
    
    if max(h, w) <= 120:
        # upscale the image_tensor with scale factor 2, bicubic
        image_tensor = torch.nn.functional.interpolate(image_tensor, scale_factor=4, mode='bicubic', align_corners=False)
        image_tensor = image_tensor.clamp(0, 1)

    if max(h, w) <= 240:
        # upscale the image_tensor with scale factor 2, bicubic
        image_tensor = torch.nn.functional.interpolate(image_tensor, scale_factor=2, mode='bicubic', align_corners=False)
        image_tensor = image_tensor.clamp(0, 1)
    
    if max(h, w) > 4200:
        # downscale the image_tensor with scale factor 0.5, bicubic
        image_tensor = torch.nn.functional.interpolate(image_tensor, scale_factor=0.5, mode='bicubic', align_corners=False)
        image_tensor = image_tensor.clamp(0, 1)

    for metric_name, metric_key in TARGET_METRICS.items():
        iqa_model = pyiqa.create_metric(metric_key, device=device)
        iqa_model.to(device)
        score = iqa_model(image_tensor).item()
        results[metric_name] = round(score, 4)
        del iqa_model
        # Release memory
        gc.collect()
        torch.cuda.empty_cache()
    
    # Define weights for each metric
    weights = {
        "CLIPIQA+": 1.0,
        "MANIQA": 1.0,
        "MUSIQ": 0.01,
        "NIQE": 1.0
    }

    # Compute weighted sum
    weighted_score = sum(
        results[metric] * weights[metric] if metric != "NIQE" else (1 - results[metric] / 10) * weights[metric]
        for metric in results
    )
    weighted_score = round(weighted_score, 4) / len(results)

    # Release memory
    del image_tensor
    torch.cuda.empty_cache()

    return weighted_score


def compute_iqa_metric_score_batch(image_paths: list[str]) -> list[float]:
    """Compute IQA metric scores for a batch of images, reusing IQA models."""
    models = {
        metric_name: pyiqa.create_metric(metric_key, device=device).to(device)
        for metric_name, metric_key in TARGET_METRICS.items()
    }

    weights = {
        "CLIPIQA+": 1.0,
        "MANIQA": 1.0,
        "MUSIQ": 0.01,
        "NIQE": 1.0
    }

    scores = []

    for image_path in image_paths:
        try:
            image_tensor = image_to_tensor(image_path)
            _, _, h, w = image_tensor.size()

            if max(h, w) <= 120:
                image_tensor = torch.nn.functional.interpolate(image_tensor, scale_factor=4, mode='bicubic', align_corners=False).clamp(0, 1)
            elif max(h, w) <= 240:
                image_tensor = torch.nn.functional.interpolate(image_tensor, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)
            elif max(h, w) > 4200:
                image_tensor = torch.nn.functional.interpolate(image_tensor, scale_factor=0.5, mode='bicubic', align_corners=False).clamp(0, 1)
                
            results = {}
            for metric_name, model in models.items():
                score = model(image_tensor).item()
                results[metric_name] = round(score, 4)
                
            weighted_score = sum(
                results[m] * weights[m] if m != "NIQE" else (1 - results[m] / 10) * weights[m]
                for m in results
            )
            weighted_score = round(weighted_score / len(results), 4)
            scores.append(weighted_score)
            
            del image_tensor
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[Warning] Failed to process {image_path}: {e}")
            scores.append(None)
            
    for model in models.values():
        del model
    torch.cuda.empty_cache()

    return scores


if __name__ == "__main__":
    image_path = "/path/to/image.png"
    print(compute_iqa(image_path))
