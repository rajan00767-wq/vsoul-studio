"""Qwen-VL comparison of a uniform candidate with both uploaded references."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


CHECKS = ("badge_present", "uniform_mismatch", "hair_changed", "face_artifacts",
          "harsh_hair_glare", "background_mismatch", "head_cropped")


def check_uniform_identity(source, candidate, minimum_similarity=0.50):
    """Check source identity without inferring gender or substituting pixels."""
    import cv2
    import numpy as np
    from pipelines.photo_restoration import _get_insight_app
    try:
        detector = _get_insight_app()
        faces = [detector.get(cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR))
                 for img in (source, candidate)]
        if any(len(items) != 1 for items in faces):
            return {"accepted": False, "reason": "Expected one face in source and generated image"}
        score = float(np.dot(faces[0][0].normed_embedding, faces[1][0].normed_embedding))
        accepted = bool(np.isfinite(score) and score >= minimum_similarity)
        return {"accepted": accepted, "similarity": score if np.isfinite(score) else None,
                "minimum_similarity": minimum_similarity,
                "reason": "identity match" if accepted else "Generated face does not match the uploaded person"}
    except Exception as exc:
        return {"accepted": False, "reason": f"Identity verification unavailable: {exc}"}


def pose_offset(keypoints):
    """Nose displacement along the eye axis, invariant to image scale/roll."""
    import numpy as np
    points = np.asarray(keypoints, dtype=float)
    eye_axis = points[1] - points[0]
    squared_distance = float(eye_axis @ eye_axis)
    if squared_distance < 1:
        raise ValueError("Face landmarks are too small for pose checking")
    return float((points[2] - (points[0] + points[1]) / 2) @ eye_axis / squared_distance)


def check_source_pose(source, candidate):
    import cv2
    import numpy as np
    from pipelines.photo_restoration import _get_insight_app
    detector = _get_insight_app()
    offsets = []
    for image in (source, candidate):
        faces = detector.get(cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR))
        if len(faces) != 1:
            return {"accepted": False, "reason": "Expected exactly one face in each image"}
        offsets.append(pose_offset(faces[0].kps))
    delta = abs(offsets[0] - offsets[1])
    return {"accepted": delta <= 0.18, "source_offset": offsets[0],
            "candidate_offset": offsets[1], "difference": delta, "maximum_difference": 0.18}


def validate_review(data):
    if not isinstance(data, dict) or any(type(data.get(key)) is not bool for key in CHECKS):
        raise ValueError("Uniform review did not return all required boolean checks")
    issues = [key for key in CHECKS if data[key]]
    correction = data.get("correction")
    if isinstance(correction, (list, dict)):
        correction = json.dumps(correction, ensure_ascii=True)
    if not isinstance(correction, str) or not correction.strip():
        correction = "Correct these observed defects: " + ", ".join(issues) if issues else ""
    return {**data, "correction": correction, "issues": issues}


def review_uniform(person, template, candidate, background_rgb):
    from pipelines.uniform_vl_analyzer import MODEL_DIR, clean_and_repair_json

    prompt = (
        "Inspect this generated portrait only. "
        f"Its required flat background RGB is {background_rgb}. "
        "Return JSON only with boolean keys: badge_present, face_artifacts, harsh_hair_glare, background_mismatch, head_cropped. "
        "badge_present: visible logo, emblem or lettering on clothing. Seams and plain pockets are NOT badges. "
        "face_artifacts: visible pixel blocks, smearing, doubled eyes or unnatural facial texture; do not judge identity. "
        "harsh_hair_glare: strong metallic blue/silver glare, not ordinary soft sheen. "
        "background_mismatch: obvious wrong background color or remnants. head_cropped: missing hair/accessories at frame edges. "
        "Also return correction: short specific editing instructions for observed defects only, with locations. "
        "Include badge_evidence describing its location and visible symbol, or none. Do not invent defects or infer hidden details."
    )
    garment_prompt = (
        "Compare image 1 (uniform template) with image 2 (generated portrait). Ignore any template badge. "
        "Return JSON: uniform_mismatch (boolean), evidence (string). Compare collar first: "
        "a standing band and folded triangular points are different. Then compare fabric hue, base/thread colors, "
        "check spacing and vest seams. Name the actual visible differences; do not assume they match."
    )
    hair_prompt = (
        "Compare image 1 (original person) with image 2 (generated portrait). Ignore clothing and background. "
        "Return JSON: hair_changed (boolean), evidence (string). Check hair shape, length, parting and "
        "accessories on each side. Different lighting alone is not a hairstyle change."
    )
    cache = Path("scratch/cache")
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="uniform_review_", dir=cache) as folder:
        paths = []
        for index, img in enumerate((person, template, candidate)):
            path = Path(folder) / f"image_{index}.png"
            preview = img.convert("RGB").copy()
            preview.thumbnail((768, 768), Image.Resampling.LANCZOS)
            preview.save(path)
            paths.append(str(path.resolve()))
        worker = Path(folder) / "worker.py"
        worker.write_text(f'''
import json
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
model_dir = {str(MODEL_DIR.resolve())!r}
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, quantization_config=config, device_map={{"": 0}}, local_files_only=True).eval()
images = [Image.open(path).convert("RGB") for path in {paths!r}]
def inspect(selected, prompt):
    content = [{{"type": "image", "image": img}} for img in selected] + [{{"type": "text", "text": prompt}}]
    text = processor.apply_chat_template([{{"role": "user", "content": content}}], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=selected, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=384, do_sample=False)
    return processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
results = [inspect([images[2]], {prompt!r}),
           inspect([images[1], images[2]], {garment_prompt!r}),
           inspect([images[0], images[2]], {hair_prompt!r})]
print("REVIEW_JSON=" + json.dumps(results))
''', encoding="utf-8")
        result = subprocess.run([sys.executable, str(worker)], capture_output=True, text=True, timeout=300)
        if result.returncode:
            raise RuntimeError("Uniform review worker failed: " + result.stderr[-800:])
        line = next((s for s in result.stdout.splitlines() if s.startswith("REVIEW_JSON=")), None)
        if line is None:
            raise RuntimeError("Uniform review returned no result")
        raw = json.loads(line.split("=", 1)[1])
        portrait, garment, hair = [clean_and_repair_json(item) for item in raw]
        portrait["uniform_mismatch"] = garment.get("uniform_mismatch")
        portrait["hair_changed"] = hair.get("hair_changed")
        portrait["garment_evidence"] = garment.get("evidence")
        portrait["hair_evidence"] = hair.get("evidence")
        return validate_review(portrait)
