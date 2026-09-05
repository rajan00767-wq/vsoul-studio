#!/usr/bin/env python3
"""Download only the Qwen models required by this Vsoul Studio workspace."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "models"

# Paths relative to this project's models/ directory.
NEEDED = [
    "qwen_image_edit/tokenizer",
    "qwen_image_edit/processor",
    "qwen_image_edit/text_encoder",
    "qwen_image_edit/vae",
    "qwen_image_edit/scheduler",
    "qwen_image_edit/qwen-image-edit-2511-Q4_K_M.gguf",
    "qwen_image_edit/svdq-fp4_r32-qwen-image-edit-lightningv1.0-4steps.safetensors",
    "loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    "qwen_image_edit/transformer/config.json",
    "qwen2.5-vl-3b-instruct",
    "realesrgan/RealESRGAN_x4plus.pth",
    "insightface",
]


def _present(rel: str) -> bool:
    path = DEST / rel
    return path.is_file() or (path.is_dir() and any(path.rglob("*")))


def fetch_missing() -> None:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("[WARN] huggingface_hub not installed; cannot fetch missing files.")
        return

    if not (DEST / "qwen_image_edit" / "qwen-image-edit-2511-Q4_K_M.gguf").is_file():
        print("[FETCH] Qwen Image Edit GGUF")
        hf_hub_download(
            "unsloth/Qwen-Image-Edit-2511-GGUF",
            "qwen-image-edit-2511-Q4_K_M.gguf",
            local_dir=str(DEST / "qwen_image_edit"),
        )

    if not (DEST / "loras" / "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors").is_file():
        print("[FETCH] Lightning LoRA")
        hf_hub_download(
            "lightx2v/Qwen-Image-Edit-2511-Lightning",
            "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            local_dir=str(DEST / "loras"),
        )

    qwen_comp = DEST / "qwen_image_edit"
    if not (qwen_comp / "vae").is_dir() or not (qwen_comp / "transformer" / "config.json").is_file():
        print("[FETCH] Qwen companion (vae/tokenizer/text_encoder/scheduler/processor)")
        snapshot_download(
            "Qwen/Qwen-Image-Edit-2511",
            local_dir=str(qwen_comp),
            allow_patterns=[
                "vae/*",
                "tokenizer/*",
                "processor/*",
                "scheduler/*",
                "text_encoder/*",
                "transformer/config.json",
                "*.json",
            ],
            ignore_patterns=["transformer/*.safetensors", "transformer/*.bin"],
        )

    esr = DEST / "realesrgan" / "RealESRGAN_x4plus.pth"
    if not esr.is_file():
        print("[FETCH] RealESRGAN")
        hf_hub_download(
            "ai-forever/Real-ESRGAN",
            "RealESRGAN_x4plus.pth",
            local_dir=str(DEST / "realesrgan"),
        )

    vl_dir = DEST / "qwen2.5-vl-3b-instruct"
    if not vl_dir.is_dir() or not any(vl_dir.iterdir()):
        print("[FETCH] Qwen2.5-VL 3B Instruct analysis model")
        snapshot_download(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            local_dir=str(vl_dir),
        )


def report() -> int:
    missing = [rel for rel in NEEDED if not _present(rel)]
    print("\n=== Model check ===")
    for rel in NEEDED:
        mark = "OK" if _present(rel) else "MISSING"
        print(f"  [{mark}] {rel}")
    return 0 if not missing else 1


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    fetch_missing()
    return report()


if __name__ == "__main__":
    sys.exit(main())
