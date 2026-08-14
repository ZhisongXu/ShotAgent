import os
import json
import logging
from typing import Union, Optional

from PIL import Image

import torch
import outlines
from outlines.models.transformers_vision import transformers_vision
from transformers import MllamaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from pydantic import BaseModel

from utils.expert_IQA_eval import compute_iqa
from pipeline import prompts

from .base_llm import BaseLLM


script_dir = os.path.dirname(os.path.abspath(__file__))

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# Optional device map that one can use to let `transformers` share a single GPU and CPU.
DEVICE_MAP = {
    "visual": 1,
    "model.embed_tokens": 1,
    "model.layers.0": 1,
    "model.layers.1": 1,
    "model.layers.2": 1,
    "model.layers.3": 1,
    "model.layers.4": 1,
    "model.layers.5": 1,
    "model.layers.6": 1,
    "model.layers.7": 1,
    "model.layers.8": 1,
    "model.layers.9": 1,
    "model.layers.10": 1,
    "model.layers.11": "cpu",
    "model.layers.12": "cpu",
    "model.layers.13": "cpu",
    "model.layers.14": "cpu",
    "model.layers.15": "cpu",
    "model.layers.16": "cpu",
    "model.layers.17": "cpu",
    "model.layers.18": "cpu",
    "model.layers.19": "cpu",
    "model.layers.20": "cpu",
    "model.layers.21": "cpu",
    "model.layers.22": "cpu",
    "model.layers.23": "cpu",
    "model.layers.24": "cpu",
    "model.layers.25": "cpu",
    "model.layers.26": "cpu",
    "model.layers.27": "cpu",
    "model.norm": "cpu",
    "model.rotary_emb": "cpu",
    "lm_head": "cpu",
}


class Score(BaseModel):
    explanation: str
    score: float


class Perception(BaseModel):
    degradations: list[str]
    tasks: list[str]
    image_description: str


class Plan(BaseModel):
    plan: list[str]


class PerceptionVLMAgent(BaseLLM):
    def __init__(
        self,
        seed: int = 1994,
        use_low_gpu_vram: bool = False,
        config_path: Optional[str] = None,
        log_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        silent: bool = False,
        system_message: Optional[str] = None,
    ):
        super().__init__(config_path, log_path, logger, silent)

        model, processor = self.load_verifier()

        # device = "cuda:0" if not use_low_gpu_vram else "cpu"
        device = "cuda" if not use_low_gpu_vram else "cpu"
        model_kwargs = {"torch_dtype": torch.bfloat16}
        # model_kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"}

        self.model = transformers_vision(
            MODEL_ID,
            model_class=model.__class__,
            device=device,
            model_kwargs=model_kwargs,
            processor_class=processor.__class__,
        )
        self.structured_generator_perception = outlines.generate.json(self.model, Perception)
        self.structured_generator_plan = outlines.generate.json(self.model, Plan)

        del model, processor

        assert system_message is not None, "system_message must be provided"
        self.perception_agent_system_message = system_message
        self.seed = seed

        self._log("**System message for Perception VLM Agent**")
        self._log(self.perception_agent_system_message)

        self.degradation_to_task = {
            "noise": "denoising",
            "motion blur": "motion deblurring",
            "defocus blur": "defocus deblurring",
            "haze": "dehazing",
            "rain": "deraining",
            "dark": "brightening",
            "jpeg compression artifact": "jpeg compression artifact removal",
        }

    @torch.no_grad()
    def load_verifier(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID)
        # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     MODEL_ID,
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto",
        # )
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        return model, processor

    def prepare_conversations(self, prompt: str):
        conversation = [
            {"role": "system", "content": self.perception_agent_system_message},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        return conversation

    def prepare_inputs(
        self,
        image_path: Union[str, list[str]],
        prompts: Union[str, list[str]],
    ) -> dict:
        if isinstance(image_path, str):
            images = [Image.open(image_path).convert("RGB")]
        elif isinstance(image_path, list):
            images = [Image.open(p).convert("RGB") for p in image_path]
        else:
            raise ValueError("image_path should be a string or a list of strings.")

        if isinstance(prompts, str):
            prompts = [prompts]

        assert len(images) == len(prompts), "Number of images and prompts must match."

        conversations = [self.prepare_conversations(p) for p in prompts]
        prompts_formatted = [self.model.processor.apply_chat_template(msg, add_generation_prompt=True) for msg in conversations]
        images_formatted = [[img] for img in images]

        inputs = {"images": images_formatted, "prompts": prompts_formatted}

        self._log("**Input of Perception VLM Agent**")
        self._log(inputs["prompts"][0])

        return inputs

    @torch.no_grad()
    def perception(self, inputs: dict, max_new_tokens: int) -> dict:
        outputs = self.structured_generator_perception(
            inputs["prompts"], inputs["images"], max_tokens=max_new_tokens, seed=self.seed
        )
        output = outputs[0].model_dump()

        print("output:", output)
        assert isinstance(output["degradations"], list) and all(isinstance(d, str) for d in output["degradations"])

        valid_degradations = [
            'noise', 'motion blur', 'defocus blur', 'haze',
            'rain', 'dark', 'jpeg compression artifact'
        ]
        output["degradations"] = [d for d in output["degradations"] if d in valid_degradations]

        self._log("**Perception result of Perception VLM Agent**")
        self._log(str(output))
        return output

    @torch.no_grad()
    def plan(self, inputs: dict, max_new_tokens: int) -> dict:
        outputs = self.structured_generator_plan(
            inputs["prompts"], inputs["images"], max_tokens=max_new_tokens, seed=self.seed
        )
        output = outputs[0].model_dump()

        self._log("**Plan result of Perception VLM Agent**")
        self._log(str(output))
        return output
