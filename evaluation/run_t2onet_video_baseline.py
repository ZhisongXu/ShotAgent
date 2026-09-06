"""Adapt the official T2ONet image editor to prompt-controlled video frames."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import string
import sys
from pathlib import Path

import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from video_retouch.io import decode_video, encode_video


def _load_model(repository: Path, device: str):
    os.chdir(repository)
    import kornia

    if not hasattr(kornia, "rgb_to_hsv"):
        kornia.rgb_to_hsv = kornia.color.rgb_to_hsv
        kornia.hsv_to_rgb = kornia.color.hsv_to_rgb
    sys.path.insert(0, str(repository))
    train_options = importlib.import_module("options.fiveK_train_options")
    actor_module = importlib.import_module("models.actor")
    options = train_options.TrainOptions()
    opt = options.parser.parse_args([])
    opt.gpu_ids = [0] if device == "cuda" else []
    opt.vocab_dir = str(repository / "data/language")
    opt.run_dir = str(repository / "output/FiveK_trial_1")
    model = actor_module.Actor(opt).to(device)
    checkpoint = repository / "output/FiveK_trial_1/seq2seqL1_model/checkpoint_best/model.pth"
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False), strict=False
    )
    return model.eval().requires_grad_(False), opt


def _vocabulary(repository: Path) -> dict[str, int]:
    path = repository / "data/language/FiveK_vocabs_sess_1.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    return {token: index for index, token in enumerate(values)}


def _tokens(prompt: str, vocabulary: dict[str, int], device: str) -> torch.Tensor:
    table = str.maketrans("", "", string.punctuation)
    words = [word.lower().translate(table) for word in prompt.split()]
    words = [word for word in words if len(word) > 1 and word.isalpha()][:15]
    ids = [vocabulary.get(word, 3) for word in words]
    sequence = ids + [0] * (15 - len(ids))
    end = sequence.index(0) if 0 in sequence else len(sequence)
    sequence.insert(end, 2)
    sequence.insert(0, 1)
    return torch.tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)


def run(
    manifest_path: Path,
    repository: Path,
    output_dir: Path,
    limit: int | None,
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload["samples"][:limit] if limit else payload["samples"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = _load_model(repository, device)
    vocabulary = _vocabulary(repository)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in samples:
        decoded = decode_video(
            (manifest_path.parent / sample["input"]).resolve(),
            max_frames=sample.get("max_frames"),
            max_side=sample.get("max_side"),
        )
        language = _tokens(str(sample["instruction"]), vocabulary, device)
        frames = []
        for frame in decoded.frames:
            tensor = pil_to_tensor(frame.convert("RGB")).float().div(255.0)
            tensor = tensor.unsqueeze(0).to(device)
            with torch.inference_mode():
                _, predictions, _, _ = model.episode_forward(
                    language, tensor, mask_dict=None, reinforce_sample=False
                )
            frames.append(to_pil_image(predictions[:, -1].squeeze(0).cpu().clamp(0, 1)))
        destination = output_dir / f"{sample['id']}.mp4"
        encode_video(frames, destination, decoded.fps, preset="veryfast")
        records.append(
            {
                "sample": sample["id"],
                "prompt": sample["instruction"],
                "frame_count": len(frames),
                "output": str(destination),
            }
        )
        print(destination, flush=True)
    (output_dir / "run.json").write_text(
        json.dumps(
            {
                "method": "T2ONet (Shi et al., CVPR 2021)",
                "adapter": "official global image editor applied framewise with fixed prompt",
                "prompt_token_policy": "official 15-word input limit; unknown vocabulary maps to UNK",
                "checkpoint_repository": "https://github.com/jshi31/T2ONet",
                "samples": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(
        args.manifest.resolve(),
        args.repository.resolve(),
        args.output_dir.resolve(),
        args.limit,
    )


if __name__ == "__main__":
    main()
