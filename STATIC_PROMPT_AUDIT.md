# Authoritative Static Single Prompt Audit (Run 20261126_132126)

## 1. Universal Static Prompt Architecture
The pipeline uses `_build_universal_static_uniform_prompt` as the single authoritative prompt builder.
It constructs a unified prompt combining 4 core constraint blocks:

1. **HEAD LOCK**:
   - `ONE PERSON ONLY. LOCAL PHOTOGRAPHIC EDIT OF IMAGE 1.`
   - `HEAD LOCK: IMAGE 1's complete head is read-only identity. Copy it visually at the exact same coordinates.`
   - Preserves exact facial bone structure, eye gaze, expression, jawline, ears, and natural photographic skin pore texture without airbrushing.
   - Identity accessories (bindi, authentic ear jewelry) explicitly guarded; earrings suppressed for masculine short-hair portraits.

2. **HAIR LOCK**:
   - `HAIR LOCK: preserve IMAGE 1's exact outer silhouette, hairline, crown, part, fringe, texture, direction, length, endpoints and every visible head item at the same coordinates.`
   - Authentic hair accessories (ribbons, bows, clips) preserved at literal coordinates.
   - Long braids preserved over chest; extra braids and artificial hair additions strictly prohibited.

3. **UNIFORM EDIT**:
   - `UNIFORM EDIT: change only clothing below the neck. IMAGE 2 is the sole design authority.`
   - Injects reference-specific garment specifications: fabric weave, collar geometry, button placket, buttons, pocket, and sleeves.
   - Collar seating calibrated to neck classification (open neckline at base of throat).

4. **CAMERA PHOTO**:
   - Flat solid studio backdrop requested in exact RGB format.
   - Lens texture and photographic realism mandated; painting, illustration, and plastic skin prohibited.

## 2. Token Budget & Truncation Verification
- Maximum positive prompt length: ~220-280 words (~320-390 tokens).
- Maximum negative prompt length: ~110-150 words (~140-190 tokens).
- Total text sequence: well within Qwen-Image-Edit 500-token context limit.
- Zero mid-sentence prompt truncation verified.
