import numpy as np
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf

from .loop import InferenceLoop, MODELS
from ..utils.common import (
    instantiate_from_config,
    load_model_from_url,
    trace_vram_usage,
)
from ..pipeline import (
    BSRNetPipeline,
    SwinIRPipeline,
)
from ..model import RRDBNet, SwinIR


class BSRInferenceLoop(InferenceLoop):

    def load_cleaner(self) -> None:
        base_dir = Path(__file__).resolve().parent.parent.parent / "configs" / "inference"

        if self.args.version == "v1":
            config_path = base_dir / "swinir.yaml"
            weight = MODELS["swinir_general"]
        elif self.args.version == "v2":
            config_path = base_dir / "bsrnet.yaml"
            weight = MODELS["bsrnet"]
        else:
            config_path = base_dir / "swinir.yaml"
            weight = MODELS["swinir_realesrgan"]

        self.cleaner: RRDBNet | SwinIR = instantiate_from_config(OmegaConf.load(config_path))
        model_weight = load_model_from_url(weight)
        self.cleaner.load_state_dict(model_weight, strict=True)
        self.cleaner.eval().to(self.args.device)

    def load_pipeline(self) -> None:
        if self.args.version == "v1" or self.args.version == "v2.1":
            self.pipeline = SwinIRPipeline(
                self.cleaner,
                self.cldm,
                self.diffusion,
                self.cond_fn,
                self.args.device,
            )
        else:
            self.pipeline = BSRNetPipeline(
                self.cleaner,
                self.cldm,
                self.diffusion,
                self.cond_fn,
                self.args.device,
                self.args.upscale,
            )

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        if self.args.version == "v1" or self.args.version == "v2.1":
            lq = lq.resize(
                tuple(int(x * self.args.upscale) for x in lq.size), Image.BICUBIC
            )
        return super().after_load_lq(lq)
