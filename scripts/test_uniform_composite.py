"""Explicit opt-in experiment; never changes the live enhancement/uniform routes."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image
from pipelines.uniform_composite import (
    build_rough_composite, describe_fabric_color, describe_hair_correction, refinement_prompt,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--steps", type=int, choices=(4, 20), default=20)
    args = parser.parse_args()
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    prefix = output / args.job
    source = Image.open(args.person).convert("RGB")
    source.thumbnail((704, 704), Image.Resampling.LANCZOS)
    from pipelines.qwen_edit_pipeline import QwenEditPipeline
    pipe = QwenEditPipeline()
    if not args.generate:
        from pipelines.photo_restoration import _get_insight_app
        from pipelines.schp_service import parse
        from pipelines.uniform_badge_cleanup import remove_template_badges
        template = Image.open(args.template).convert("RGBA")
        clean, count = remove_template_badges(template, template)
        clean = clean.convert("RGBA")
        clean.putalpha(template.getchannel("A"))
        faces = _get_insight_app().get(cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR))
        if len(faces) != 1:
            raise ValueError("Exactly one source face is required")
        labels = parse(np.array(source))["labels"]
        rough, placement = build_rough_composite(source, clean, labels, faces[0].bbox, (4, 126, 246))
        rough.save(str(prefix) + "_rough.png")
        reference = pipe._prepare_uniform_reference(clean, pad_to_portrait=False)
        reference.save(str(prefix) + "_reference.png")
        audit = {"source": args.person, "template": args.template, "placement": placement,
                 "template_badges_removed": count, "steps": args.steps, "seed": 405353848,
                 "prompt": refinement_prompt((4, 126, 246), describe_fabric_color(clean),
                                             describe_hair_correction("black")),
                 "background": [4, 126, 246]}
        Path(str(prefix) + "_test.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print("Rough composite ready:", str(prefix) + "_rough.png", flush=True)
        return
    audit = json.loads(Path(str(prefix) + "_test.json").read_text(encoding="utf-8"))
    result = pipe.qwen_edit_enhancer(
        image_input=Image.open(str(prefix) + "_rough.png").convert("RGB"),
        reference_images=[Image.open(str(prefix) + "_reference.png").convert("RGB")],
        prompt=audit["prompt"], negative_prompt="different person, changed hairstyle, badge, logo, lettering, plastic fabric, wrong pattern, glossy cloth, distorted collar, duplicate face, extra hands",
        job_id=args.job, steps=args.steps, seed=audit["seed"], background_color="4,126,246",
        width=600, height=800, timeout_seconds=None, max_sequence_length=384,
        progress_callback=lambda p, m: print(f"{p}% {m}", flush=True),
        preserve_source_clothing=False, max_generation_dimension=704, minimum_generation_dimension=704,
        keep_generation_resolution=True, use_birefnet_background=False, true_cfg_scale=3.0,
        face_lock_after_generation=False, reject_identity_failure=False, return_raw_candidate=True,
        conditioning_max_dimension=384, conditioning_pixel_budget=384 * 384, preserve_reference_aspect=True)
    from pipelines.uniform_review import check_uniform_identity
    review = check_uniform_identity(source, Image.open(result).convert("RGB"))
    audit.update(raw_output=str(result), identity_review=review)
    Path(str(prefix) + "_test.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("Raw result:", result, "Review:", review, flush=True)


if __name__ == "__main__":
    main()
