"""Provider-neutral vision-language clients for multi-model orchestration."""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from typing import Protocol, Sequence

from PIL import Image

from .tasks import AGENT_SYSTEM_PROMPT


def _progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


class VisionJSONDecodeError(ValueError):
    def __init__(self, message: str, candidate: str) -> None:
        super().__init__(message)
        self.candidate = candidate


class VisionLanguageClient(Protocol):
    model_id: str

    def generate_json(
        self,
        labeled_images: Sequence[tuple[str, Image.Image]],
        prompt: str,
    ) -> dict[str, object]: ...


def extract_json_object(text: str) -> dict[str, object]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0:
        raise ValueError(f"Vision model response did not contain JSON: {text[:240]}")
    if end <= start:
        raise VisionJSONDecodeError(
            f"Vision model response contained incomplete JSON: {text[:240]}",
            text[start:],
        )
    candidate = text[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        excerpt_start = max(0, error.pos - 160)
        excerpt_end = min(len(candidate), error.pos + 160)
        excerpt = candidate[excerpt_start:excerpt_end]
        raise VisionJSONDecodeError(
            "Vision model response contained invalid JSON "
            f"near character {error.pos}: {excerpt}",
            candidate,
        ) from error
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
        if self.max_image_side < 64:
            raise ValueError("max_image_side must be at least 64 pixels.")

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
        body = json.dumps(
            {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "temperature": 0,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()
        _progress(
            f"calling chat model={self.model_id} images={len(labeled_images)}"
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[-2000:]
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
        _progress(
            f"chat model={self.model_id} finished in {time.monotonic() - started:.1f}s"
        )
        return extract_json_object(text)


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
    ) -> None:
        super().__init__(
            base_url=base_url,
            model_id=model_id,
            api_key_env=api_key_env,
            timeout_seconds=timeout_seconds,
            max_tokens=max_output_tokens,
            jpeg_quality=jpeg_quality,
            max_image_side=max_image_side,
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

    def _post_responses(
        self,
        api_key: str,
        request_payload: dict[str, object],
        *,
        label: str,
        images: int,
    ) -> dict[str, object]:
        body = json.dumps(request_payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(3):
            started = time.monotonic()
            _progress(
                f"calling {label} model={self.model_id} "
                f"reasoning={self.reasoning_effort} detail={self.image_detail} "
                f"images={images} attempt={attempt + 1}/3"
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                _progress(
                    f"{label} model={self.model_id} finished in "
                    f"{time.monotonic() - started:.1f}s"
                )
                return payload
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[-2000:]
                if error.code != 429 and error.code < 500:
                    raise RuntimeError(
                        f"Responses API returned HTTP {error.code}: {detail}"
                    ) from error
                if attempt == 2:
                    raise RuntimeError(
                        f"Responses API returned HTTP {error.code} after "
                        f"3 attempts: {detail}"
                    ) from error
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 2.0**attempt
                _progress(
                    f"responses HTTP {error.code}; retrying after {delay:.1f}s"
                )
                time.sleep(min(max(delay, 0.0), 30.0))
        raise RuntimeError("Responses API retry loop exited unexpectedly.")

    def _repair_json(self, api_key: str, broken_json: str) -> dict[str, object]:
        _progress("repairing invalid JSON with text-only Responses request")
        repair_prompt = (
            "Repair this malformed JSON into one valid JSON object. Preserve all "
            "keys, arrays, numbers, booleans, and strings as much as possible. "
            "Do not add explanation. Return JSON only.\n\n"
            f"{broken_json}"
        )
        request_payload: dict[str, object] = {
            "model": self.model_id,
            "instructions": "You repair malformed JSON for a video pipeline.",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": repair_prompt}],
                }
            ],
            "max_output_tokens": max(self.max_tokens, 4096),
            "text": {"format": {"type": "json_object"}},
        }
        if self.reasoning_effort != "none":
            request_payload["reasoning"] = {"effort": self.reasoning_effort}
        payload = self._post_responses(
            api_key,
            request_payload,
            label="responses-json-repair",
            images=0,
        )
        return extract_json_object(self._response_text(payload))

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
            "text": {"format": {"type": "json_object"}},
        }
        if self.reasoning_effort != "none":
            request_payload["reasoning"] = {"effort": self.reasoning_effort}
        payload = self._post_responses(
            api_key,
            request_payload,
            label="responses",
            images=len(labeled_images),
        )
        text = self._response_text(payload)
        try:
            return extract_json_object(text)
        except VisionJSONDecodeError as error:
            return self._repair_json(api_key, error.candidate)
