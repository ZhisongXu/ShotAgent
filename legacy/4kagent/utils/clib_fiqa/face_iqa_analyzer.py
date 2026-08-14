import torch
import torch.nn.functional as F
from itertools import product
from PIL import Image
import torchvision.transforms as T
from .model import clip
from .utilities import load_net_param, dist_to_score


class CLIBFIQAScorer:
    """
    A class to analyze face image attributes and quality using a CLIP-based model.
    """

    blur_list = ['hazy', 'blurry', 'clear']
    occ_list = ['obstructed', 'unobstructed']
    pose_list = ['profile', 'slight angle', 'frontal']
    exp_list = ['exaggerated expression', 'typical expression']
    ill_list = ['extreme lighting', 'normal lighting']
    quality_list = ['bad', 'poor', 'fair', 'good', 'perfect']

    pose_map = {i: pose for i, pose in enumerate(pose_list)}
    blur_map = {i: blur for i, blur in enumerate(blur_list)}
    occ_map  = {i: occ for i, occ in enumerate(occ_list)}
    ill_map  = {i: ill for i, ill in enumerate(ill_list)}
    exp_map  = {i: exp for i, exp in enumerate(exp_list)}

    def __init__(self, clip_model_path: str, clip_weights_path: str, device: str = 'cuda'):
        """
        Initialize the analyzer by loading the CLIP model and weights, and building the joint text embeddings.
        """
        self.device = device
        # Load CLIP backbone
        self.model, _ = clip.load(clip_model_path, device=device, jit=False)
        # Load fine-tuned weights
        self.model = load_net_param(self.model, clip_weights_path)
        self.model.eval()

        # Prepare joint text prompts
        texts = []
        for b, o, p, e, l, q in product(
            self.blur_list,
            self.occ_list,
            self.pose_list,
            self.exp_list,
            self.ill_list,
            self.quality_list
        ):
            prompt = f"a photo of a {b}, {o}, and {p} face with {e} under {l}, which is of {q} quality"
            texts.append(prompt)

        # Tokenize and move to device
        self.joint_texts = torch.cat([clip.tokenize(t) for t in texts]).to(device)

        # Image preprocessing pipeline
        self.transform = T.Compose([
            T.Resize([224, 224]),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess a single image."""
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img)
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def analyze(self, image_path: str) -> dict:
        """
        Analyze the given image and return predicted attributes and quality.

        Returns:
            dict: {
                'blur': str,
                'occlusion': str,
                'pose': str,
                'expression': str,
                'lighting': str,
                'quality_score': float
            }
        """
        # Preprocess
        img_tensor = self._preprocess_image(image_path)
        batch_size = img_tensor.size(0)

        # Forward pass
        logits_image, _ = self.model.forward(img_tensor.view(-1, 3, 224, 224), self.joint_texts)
        logits_image = F.softmax(logits_image.view(batch_size, -1), dim=1)

        # Reshape to separate dimensions
        dims = (
            len(self.blur_list),
            len(self.occ_list),
            len(self.pose_list),
            len(self.exp_list),
            len(self.ill_list),
            len(self.quality_list)
        )
        logits_image = logits_image.view(batch_size, *dims)

        # Quality distribution and score
        quality_dist = logits_image.sum(1).sum(1).sum(1).sum(1).sum(1)
        quality_score = dist_to_score(quality_dist).item()

        # Max predictions for each factor
        def argmax_over_axes(tensor, axes):
            dims = list(range(1, tensor.ndim))
            for ax in sorted(axes, reverse=True):
                tensor = tensor.sum(dim=ax)
            return tensor.argmax(dim=1).cpu().item()

        blur_idx = argmax_over_axes(logits_image, axes=[2,3,4,5,6])
        occ_idx  = argmax_over_axes(logits_image, axes=[1,3,4,5,6])
        pose_idx = argmax_over_axes(logits_image, axes=[1,2,4,5,6])
        exp_idx  = argmax_over_axes(logits_image, axes=[1,2,3,5,6])
        ill_idx  = argmax_over_axes(logits_image, axes=[1,2,3,4,6])

        # return {
        #     'blur': self.blur_map[blur_idx],
        #     'occlusion': self.occ_map[occ_idx],
        #     'pose': self.pose_map[pose_idx],
        #     'expression': self.exp_map[exp_idx],
        #     'lighting': self.ill_map[ill_idx],
        #     'quality_score': quality_score
        # }
        return quality_score


if __name__ == "__main__":
    analyzer = CLIBFIQAScorer(
        clip_model_path="./weights/RN50.pt",
        clip_weights_path="./weights/CLIB-FIQA_R50.pth"
    )
    result = analyzer.analyze("./Lincoln.png")
    print(result)
