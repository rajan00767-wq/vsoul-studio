# V13.3.3 Root-Cause Analysis Report: Anatomical Garment Registration & Compositing Breakdown

**Date**: 2026-11-27  
**Branch**: `v13.3-production-stabilization`  
**Target**: V13.3.3 Architectural Redesign  
**Status**: INVESTIGATION COMPLETE · ALL VISUAL REGRESSIONS DEMONSTRATED WITH PIXEL EVIDENCE  

---

## 1. Executive Summary & Visual Audit Findings

Independent visual inspection of the V13.3.2 outputs confirmed that despite achieving 100% automated quality gate delivery (44/44 passing conditions), all seven production photographs exhibited severe aesthetic and geometric defects:

- **PA_01**: Red collar mostly erased, unnatural neckline, dark shoulder outline.
- **PA_02**: Rectangular neck patch remains visible; navy collar partially erased.
- **PA_03**: Left maroon bow largely missing, rectangular chest seam, missing tie knot, black shoulder outlines.
- **PA_04**: Neckline unnatural, shoulder boundary and collar integration disrupted.
- **PA_05**: Red collar partially erased, unnatural neck integration, shoulder artifacts.
- **PA_11**: Blue hair halo improved, but black shoulder outlines and chin seam remain.
- **PA_12**: Red collar incomplete, unnatural neckline, lower sleeve/background artifacts.

Comparative inspection between the **raw Qwen candidates** (`raw_candidate_resized.png`) and the **composited deliverables** (`*_uniform.png`) revealed that the raw Qwen candidates generated intact, highly realistic uniform collars, ties, plackets, and shoulder geometry. The defects were introduced entirely by the downstream compositing pipeline (`identity_extractor.py`, `compositor.py`, `image_normalizer.py`).

---

## 2. Anatomical Scale & Coordinate Discrepancy (The Primary Root Cause)

### A. The Registration Void
The previous compositing pipeline (`compositor.py`) operated under the fatal assumption that the normalized source image (`02_normalized_source.png`) and the raw Qwen generation (`raw_candidate_resized.png`) shared identical 1:1 pixel coordinates for the face, chin, neck, and shoulders.

Quantitative measurement using InsightFace facial landmarks on all 7 cases reveals massive scale and coordinate divergence:

| Case ID | Source Chin ($y$) | Raw Chin ($y$) | $\Delta y_{\text{chin}}$ (px) | Source Face Height | Raw Face Height | Scale Ratio ($H_{\text{raw}} / H_{\text{src}}$) | Collar & Knot Overlap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **PA_01** | **712** | **529** | **-183 px** | 493 px | 366 px | **0.744** | Source chin lands 183 px into Qwen collar/chest |
| **PA_02** | **599** | **508** | **-91 px** | 460 px | 370 px | **0.804** | Source chin lands 91 px into Qwen navy collar |
| **PA_03** | **639** | **506** | **-133 px** | 499 px | 386 px | **0.774** | Source chin lands directly over Qwen tie knot |
| **PA_04** | **628** | **508** | **-120 px** | 487 px | 385 px | **0.790** | Source neck overlaps Qwen vest opening |
| **PA_05** | **730** | **523** | **-207 px** | 491 px | 358 px | **0.728** | Source chin lands 207 px into Qwen red collar |
| **PA_11** | **662** | **662** | **0 px** | 435 px | 449 px | **1.031** | Geometry aligned (~1:1); collar preserved |
| **PA_12** | **690** | **535** | **-155 px** | 491 px | 367 px | **0.748** | Source chin lands 155 px into Qwen red collar |

### B. The Consequence: Collar and Tie Obliteration
In `compositor.py` (lines 161–164 and 311–315):
```python
cand_head = (cand_labels == 13) | head_core_m
cand_head_dil = cv2.dilate(cand_head.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
cand_garment_zone &= ~cand_head_dil
...
fg_rgb, fg_alpha = _blend_straight_rgba(fg_rgb, fg_alpha, src_rgb_to_use, face_alpha)
```
Because `head_core_m` came from the source image where the face is 25% larger and the chin reaches down to $y = 712\text{--}730$, `cand_head_dil` marked everything from $y=236$ down to $y=736$ as "head".
1. In `raw_candidate_resized.png`, Qwen generated the collar starting at $y=492$.
2. The dilated source head mask erased the candidate garment between $y=492$ and $y=736$.
3. When `face_alpha` was layered, the source chin and throat skin were stamped directly on top of the uniform chest.
4. In PA_01, this erased 43,305 red collar pixels.
5. In PA_03, Qwen's tie knot was located at $y=520\text{--}600$. Stamping the source chin at $y=639$ completely crushed the tie knot into oblivion.
6. To fill the resulting gap between the source torso and the remaining garment, `compositor.py` executed Telea inpainting across the body (`body_gap`), creating horizontal seam artifacts.

---

## 3. Black Shoulder Outlines: Exact Mechanism & Demonstration

### A. The Symptom
Delivered images for PA_01, PA_03, and PA_11 exhibit thick dark/black pixels along the silhouette boundary of the shoulders:
- PA_03: 1,808 pure black pixels ($RGB < [80, 80, 80]$) along the shoulder edge.
- PA_11: 2,068 pure black pixels ($RGB = [0, 0, 0]$) along the shoulder edge.

### B. Root-Cause Operation 1: Uninitialized RGB in `_blend_straight_rgba` & Forced Alpha
In `compositor.py` (lines 46–50):
```python
out_rgb = np.zeros_like(base_rgb, dtype=np.float32)
out_rgb[pos] = (top_rgb * ao + base_rgb * ab * (1.0 - ao)) / a_out
```
Wherever $a_{\text{out}} \le 0.001$, `out_rgb` remains `[0, 0, 0]`.
Then at lines 349–350:
```python
solid_interior = face_core | (cand_garment_zone & (cand_matte > 200))
fg_alpha[solid_interior] = 255
```
This forced `fg_alpha = 255` onto boundary pixels without setting their RGB channels, locking pure black `[0, 0, 0]` into the opaque foreground!

### C. Root-Cause Operation 2: Black Sampling during Inpainting
In `compositor.py` (lines 353–373):
Internal alpha holes on the torso were inpainted with:
```python
fg_rgb = cv2.inpaint(fg_rgb, hole_fa_m, inpaintRadius=11, flags=cv2.INPAINT_TELEA)
```
Because the outer transparent region of `fg_rgb` was initialized to `[0, 0, 0]`, the Telea algorithm sampled from adjacent black pixels, bleeding solid black into edge boundaries.

### D. Root-Cause Operation 3: Black Padding in Passport Crop Resampling
In `image_normalizer.py` (line 724):
```python
padded_canvas = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
padded_canvas.paste(image, (pad_left, pad_top))
cropped_resized_pil = cropped_pil.resize(output_size, Image.Resampling.LANCZOS)
```
Padding with `(0, 0, 0, 0)` surrounds the subject with transparent black. Lanczos resampling filters across this boundary, blending subject colors with `(0, 0, 0)` and creating a dark contour fringe when composited over white.

---

## 4. Accessory Preservation Failures (PA_03 & PA_04)

### A. PA_03 Maroon Bows
- In `03_detected_accessory_objects.json`, two bow instances were detected on the source portrait:
  - `right_bow_tail` (area 5,247 px)
  - `left_bow_tail` (area 3,662 px)
- In the final composite, the left bow was largely destroyed because:
  1. It sat at $y=680\text{--}850$ on the left chest.
  2. The unaligned torso cutout and candidate garment boundary (`cand_garment_zone &= ~cand_head_dil`) treated it as background or erased it during body gap closure.
  3. No explicit instance-level geometry correspondence tracked the left bow to ensure its complete shape (loops + tails) survived through compositing.

### B. PA_04 Misleading Filename & Missed Accessories
- PA_04 was named `PA_04_NoBows_Vest.png`, but the source photograph actually contains an ornate **floral hair bow** and **earrings**.
- The pipeline previously used filename hints and global heuristics (`has_confirmed_bows`) which biased accessory thresholds.
- Instance-level extraction must operate purely on computer vision features (spatial connectivity to hair, chromatic delta, BiRefNet alpha), completely agnostic to filenames or metadata.

---

## 5. Architectural Solution Required for V13.3.3

1. **Target Garment Registration**:
   - Extract garment components (collar lapels, tie knot, placket, shoulders) from the raw Qwen candidate.
   - Compute similarity / affine transformation $T_{\text{raw}\to\text{src}}$ using face landmarks (eyes, nose, mouth, chin).
   - Warp target garment components onto source anatomy so the collar opening aligns naturally with the source chin and neck.
2. **Dynamic Semantic Neck Extraction**:
   - Visible neck skin must be defined dynamically as the intersection between authentic source neck skin and the *registered collar opening*.
   - Never allow rectangular or parabolic neck blocks to overwrite target collar lapels or tie knots.
3. **Occlusion-Aware Layer Compositing**:
   - Layer 1: Registered Target Garment (placket, buttons, shoulders).
   - Layer 2: Visible Neck Skin (in collar opening only).
   - Layer 3: Collar Lapels & Tie Knot (overlapping neck skin).
   - Layer 4: Source Face Core & Ears (straight RGBA, 100% authentic pixels).
   - Layer 5: Source Scalp Hair.
   - Layer 6: Source Braids & Long Hair (draping over garment).
   - Layer 7: Source Accessories (both bows, ribbons, clips, floral pins).
4. **Boundary Decontamination & Color Bleed**:
   - Bleed foreground RGB into transparent pixels prior to resampling.
   - Pad with `(target_bg_color, 0)` instead of `(0, 0, 0, 0)`.
   - Eliminate all black contour drawing and black-sampling inpainting.
