import os
import re
import json
import logging
from typing import Union, Optional

import torch
from PIL import Image
from pydantic import BaseModel

from transformers import MllamaForConditionalGeneration, AutoProcessor

from utils.expert_IQA_eval import compute_iqa
from pipeline import prompts
from .base_llm import BaseLLM


script_dir = os.path.dirname(os.path.abspath(__file__))

# MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MODEL_ID = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"  # for low GPU vram


class Score(BaseModel):
    explanation: str
    score: float


class Perception(BaseModel):
    degradations: list[str]
    tasks: list[str]
    image_description: str


class Plan(BaseModel):
    plan: list[str]


class LlamaVisionAgent(BaseLLM):
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

        self.model, self.processor = self.load_agent()
        self.perception_agent_system_message = system_message
        self.seed = seed
        assert self.perception_agent_system_message is not None, "System message must be provided."

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
    def load_agent(self):
        model = MllamaForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, 
            # device_map="auto",
            device_map = {"": 0}
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        return model, processor

    def prepare_conversations(self, prompt: str) -> dict:
        user_content = [{"type": "image"}, {"type": "text", "text": prompt}]
        return {"role": "user", "content": user_content}

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

        assert len(images) == len(prompts), "The number of images and prompts must match."

        conversations = [self.prepare_conversations(p) for p in prompts]

        prompt = self.processor.apply_chat_template([conversations[0]], add_generation_prompt=True)
        image = images[0]

        inputs = self.processor(image, prompt, add_special_tokens=False, return_tensors="pt").to(self.model.device)

        self._log("**Text Input of Perception LlamaVisionAgent Agent**")
        self._log(prompt)

        return inputs

    @torch.no_grad()
    def perception(self, inputs, max_new_tokens: int) -> dict:
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True)
        generated_tokens = output.sequences
        perception_output = self.processor.decode(generated_tokens[0], skip_special_tokens=True)

        json_pattern = re.compile(r'{.*?}', re.DOTALL)
        json_match = json_pattern.search(perception_output)

        degradations = []
        image_description = "An image"

        if json_match:
            json_str = json_match.group(0)
            try:
                perception_dict = json.loads(json_str)
                degradations = perception_dict.get("degradations", [])
                image_description = perception_dict.get("image_description", "")
                print("Degradations:", degradations)
                print("Image Description:", image_description)
            except json.JSONDecodeError as e:
                print("JSON ERROR:", e)

        assert isinstance(degradations, list) and all(isinstance(d, str) for d in degradations)

        self._log("**Perception result of Perception LlamaVisionAgent Agent**")
        self._log(str(degradations))
        self._log(str(image_description))

        return {
            "degradations": degradations,
            "image_description": image_description,
        }

    @torch.no_grad()
    def plan(self, inputs, agenda: list[str], max_new_tokens: int) -> dict:
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True)
        generated_tokens = output.sequences
        perception_output = self.processor.decode(generated_tokens[0], skip_special_tokens=True)

        json_pattern = re.compile(r'{.*?}', re.DOTALL)
        json_match = json_pattern.search(perception_output)

        final_plan = agenda

        if json_match:
            json_str = json_match.group(0)
            try:
                plan_dict = json.loads(json_str)
                init_plan = plan_dict.get("plan", [])
                if not isinstance(init_plan, list):
                    final_plan = json.loads(init_plan.replace("'", '"'))
                else:
                    final_plan = init_plan
                print("final_plan:", final_plan)
            except json.JSONDecodeError as e:
                print("JSON ERROR:", e)

        assert isinstance(final_plan, list) and all(isinstance(task, str) for task in final_plan)

        self._log("**Plan result of Perception LlamaVisionAgent Agent**")
        self._log(str(final_plan))

        del output
        torch.cuda.empty_cache()

        return {"plan": final_plan}
