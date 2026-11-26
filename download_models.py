#!/usr/bin/env python3
"""Download only the model assets used by this Vsoul Studio workspace."""

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
    "loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    "loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
    "qwen2.5-vl-3b-instruct",
    "birefnet-hr-matting/config.json",
    "birefnet-hr-matting/model.safetensors",
    "schp-lip-20/config.json",
    "schp-lip-20/preprocessor_config.json",
    "schp-lip-20/model.safetensors",
    "realesrgan/RealESRGAN_x4plus.pth",
    "insightface/models/buffalo_l/2d106det.onnx",
    "insightface/models/buffalo_l/det_10g.onnx",
    "insightface/models/buffalo_l/1k3d68.onnx",
    "insightface/models/buffalo_l/genderage.onnx",
    "insightface/models/buffalo_l/w600k_r50.onnx",
    "insightface/inswapper_128.onnx",
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
        print("[FETCH] 4-step Lightning LoRA")
        hf_hub_download(
            "lightx2v/Qwen-Image-Edit-2511-Lightning",
            "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            local_dir=str(DEST / "loras"),
        )

    if not (DEST / "loras" / "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors").is_file():
        print("[FETCH] 8-step Lightning LoRA")
        hf_hub_download(
            "lightx2v/Qwen-Image-Edit-2511-Lightning",
            "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
            local_dir=str(DEST / "loras"),
        )

    qwen_comp = DEST / "qwen_image_edit"
    qwen_components = ("vae", "tokenizer", "processor", "scheduler", "text_encoder")
    if any(not (qwen_comp / component).is_dir() for component in qwen_components):
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
                "*.json",
            ],
            ignore_patterns=["transformer/*"],
        )

    birefnet_dir = DEST / "birefnet-hr-matting"
    if not (birefnet_dir / "model.safetensors").is_file():
        print("[FETCH] BiRefNet HR portrait matting")
        snapshot_download(
            "ZhengPeng7/BiRefNet_HR-matting",
            local_dir=str(birefnet_dir),
            ignore_patterns=["*.md", "assets/*"],
        )

    schp_dir = DEST / "schp-lip-20"
    if not (schp_dir / "model.safetensors").is_file():
        print("[FETCH] SCHP LIP-20 human parsing")
        snapshot_download(
            "pirocheto/schp-lip-20",
            local_dir=str(schp_dir),
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
            ],
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

    buffalo_dir = DEST / "insightface" / "models" / "buffalo_l"
    buffalo_files = (
        "2d106det.onnx",
        "det_10g.onnx",
        "1k3d68.onnx",
        "genderage.onnx",
        "w600k_r50.onnx",
    )
    if any(not (buffalo_dir / f).is_file() for f in buffalo_files):
        print("[FETCH] InsightFace buffalo_l pack")
        buffalo_dir.mkdir(parents=True, exist_ok=True)
        import urllib.request
        import zipfile

        zip_url = "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l.zip"
        temp_zip = DEST / "insightface" / "buffalo_l.zip"
        try:
            urllib.request.urlretrieve(zip_url, temp_zip)
            with zipfile.ZipFile(temp_zip, "r") as z:
                z.extractall(buffalo_dir)
            temp_zip.unlink(missing_ok=True)
        except Exception as e:
            print(f"[WARN] Failed to download buffalo_l from HF: {e}")
            github_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
            try:
                urllib.request.urlretrieve(github_url, temp_zip)
                with zipfile.ZipFile(temp_zip, "r") as z:
                    z.extractall(buffalo_dir)
                temp_zip.unlink(missing_ok=True)
            except Exception as e2:
                print(f"[ERROR] Failed to download buffalo_l: {e2}")

    inswapper = DEST / "insightface" / "inswapper_128.onnx"
    if not inswapper.is_file():
        print("[FETCH] InsightFace inswapper_128.onnx")
        inswapper.parent.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(
                "ezioruan/inswapper_128.onnx",
                "inswapper_128.onnx",
                local_dir=str(DEST / "insightface"),
            )
        except Exception as e:
            print(f"[WARN] Failed to download inswapper_128 via HF hub: {e}")


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
