"""Provider-neutral vision-language clients for multi-model orchestration."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Protocol, Sequence

from PIL import Image

from .tasks import AGENT_SYSTEM_PROMPT


class VisionLanguageClient(Protocol):
    model_id: str

    def generate_json(
        self,
        labeled_images: Sequence[tuple[str, Image.Image]],
        prompt: str,
    ) -> dict[str, object]: ...


def extract_json_object(text: str) -> dict[str, object]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"Vision model response did not contain JSON: {text[:240]}")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Vision model JSON response must be an object.")
    return payload


class OpenAICompatibleVisionClient:
    """Call any OpenAI-compatible multimodal chat endpoint.

    Different instances can point to different providers and models for
    storyboard perception, Anchor planning, and critique. Secrets are read from
    environment variables and are never stored in the grade graph.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key_env: str,
        timeout_seconds: float = 180.0,
        max_tokens: int = 1024,
        jpeg_quality: int = 88,
        max_image_side: int = 1280,
        max_retries: int = 0,
        retry_base_seconds: float = 2.0,
        retry_max_seconds: float = 60.0,
        json_mode: bool = False,
        max_json_retries: int = 0,
    ) -> None:
        if not base_url or not model_id or not api_key_env:
            raise ValueError("Vision endpoint, model id, and API-key env are required.")
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.api_key_env = api_key_env
        self.timeout_seconds = float(timeout_seconds)
        self.max_tokens = int(max_tokens)
        self.jpeg_quality = int(jpeg_quality)
        self.max_image_side = int(max_image_side)
        self.max_retries = int(max_retries)
        self.retry_base_seconds = float(retry_base_seconds)
        self.retry_max_seconds = float(retry_max_seconds)
        self.json_mode = bool(json_mode)
        self.max_json_retries = int(max_json_retries)
        if self.max_image_side < 64:
            raise ValueError("max_image_side must be at least 64 pixels.")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative.")
        if self.retry_base_seconds <= 0:
            raise ValueError("retry_base_seconds must be positive.")
        if self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError(
                "retry_max_seconds must be at least retry_base_seconds."
            )
        if self.max_json_retries < 0:
            raise ValueError("max_json_retries cannot be negative.")

    def _retry_delay(
        self,
        error: urllib.error.HTTPError,
        detail: str,
        attempt: int,
    ) -> float:
        delay = self.retry_base_seconds * (2**attempt)
        retry_after = error.headers.get("Retry-After")
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        match = re.search(
            r'"retryDelay"\s*:\s*"([0-9]+(?:\.[0-9]+)?)s"', detail
        )
        if match is None:
            match = re.search(
                r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)s", detail, re.IGNORECASE
            )
        if match is not None:
            delay = max(delay, float(match.group(1)))
        return min(delay, self.retry_max_seconds)

    def _data_url(self, image: Image.Image) -> str:
        image = image.convert("RGB")
        scale = min(1.0, self.max_image_side / max(image.size))
        if scale < 1.0:
            image = image.resize(
                (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def generate_json(
        self,
        labeled_images: Sequence[tuple[str, Image.Image]],
        prompt: str,
    ) -> dict[str, object]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key environment variable: {self.api_key_env}"
            )
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        for label, image in labeled_images:
            content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._data_url(image)},
                }
            )
        messages: list[dict[str, object]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        for json_attempt in range(self.max_json_retries + 1):
            request_payload: dict[str, object] = {
                "model": self.model_id,
                "messages": messages,
                "temperature": 0,
                "max_tokens": self.max_tokens,
            }
            if self.json_mode:
                request_payload["response_format"] = {"type": "json_object"}
            body = json.dumps(request_payload).encode("utf-8")
            for attempt in range(self.max_retries + 1):
                request = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=body,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(
                        request, timeout=self.timeout_seconds
                    ) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    break
                except urllib.error.HTTPError as error:
                    detail = error.read().decode("utf-8", errors="replace")[-2000:]
                    retryable = error.code == 429 or 500 <= error.code < 600
                    if retryable and attempt < self.max_retries:
                        time.sleep(self._retry_delay(error, detail, attempt))
                        continue
                    raise RuntimeError(
                        f"Vision endpoint returned HTTP {error.code}: {detail}"
                    ) from error
            try:
                text = payload["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                raise ValueError(
                    "Vision endpoint returned an invalid chat response."
                ) from error
            if isinstance(text, list):
                text = "".join(
                    str(item.get("text", ""))
                    for item in text
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            if not isinstance(text, str) or not text:
                raise ValueError("Vision endpoint response content must be text.")
            try:
                return extract_json_object(text)
            except (json.JSONDecodeError, ValueError):
                if json_attempt >= self.max_json_retries:
                    raise
                messages = [
                    *messages,
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            "The previous response was not valid JSON. Return the "
                            "same answer as exactly one syntactically valid JSON "
                            "object, with double-quoted keys and no commentary."
                        ),
                    },
                ]
        raise AssertionError("JSON retry loop terminated unexpectedly.")


class OpenAIResponsesVisionClient(OpenAICompatibleVisionClient):
    """Call OpenAI's native Responses API with ordered image inputs."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key_env: str,
        timeout_seconds: float = 180.0,
        max_output_tokens: int = 4096,
        jpeg_quality: int = 88,
        max_image_side: int = 1280,
        reasoning_effort: str = "high",
        image_detail: str = "high",
        max_retries: int = 0,
        retry_base_seconds: float = 2.0,
        retry_max_seconds: float = 60.0,
        json_mode: bool = False,
        max_json_retries: int = 0,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model_id=model_id,
            api_key_env=api_key_env,
            timeout_seconds=timeout_seconds,
            max_tokens=max_output_tokens,
            jpeg_quality=jpeg_quality,
            max_image_side=max_image_side,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
            json_mode=json_mode,
            max_json_retries=max_json_retries,
        )
        if reasoning_effort not in {
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("Unsupported Responses API reasoning effort.")
        if image_detail not in {"low", "high", "original", "auto"}:
            raise ValueError("Unsupported Responses API image detail.")
        self.reasoning_effort = reasoning_effort
        self.image_detail = image_detail

    @staticmethod
    def _response_text(payload: object) -> str:
        if not isinstance(payload, dict):
            raise ValueError("Responses API returned a non-object payload.")
        direct = payload.get("output_text")
        if isinstance(direct, str) and direct:
            return direct
        fragments: list[str] = []
        output = payload.get("output", [])
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                content = item.get("content", [])
                if not isinstance(content, list):
                    continue
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "output_text"
                        and isinstance(part.get("text"), str)
                    ):
                        fragments.append(str(part["text"]))
        text = "".join(fragments)
        if not text:
            raise ValueError("Responses API response did not contain output text.")
        return text

    def generate_json(
        self,
        labeled_images: Sequence[tuple[str, Image.Image]],
        prompt: str,
    ) -> dict[str, object]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key environment variable: {self.api_key_env}"
            )
        content: list[dict[str, object]] = [
            {"type": "input_text", "text": prompt}
        ]
        for label, image in labeled_images:
            content.append({"type": "input_text", "text": label})
            content.append(
                {
                    "type": "input_image",
                    "image_url": self._data_url(image),
                    "detail": self.image_detail,
                }
            )
        request_payload: dict[str, object] = {
            "model": self.model_id,
            "instructions": AGENT_SYSTEM_PROMPT,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": self.max_tokens,
        }
        if self.reasoning_effort != "none":
            request_payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.json_mode:
            request_payload["text"] = {"format": {"type": "json_object"}}
        body = json.dumps(request_payload).encode("utf-8")
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/responses",
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[-2000:]
                retryable = error.code == 429 or 500 <= error.code < 600
                if retryable and attempt < self.max_retries:
                    time.sleep(self._retry_delay(error, detail, attempt))
                    continue
                raise RuntimeError(
                    f"Responses API returned HTTP {error.code}: {detail}"
                ) from error
        return extract_json_object(self._response_text(payload))
