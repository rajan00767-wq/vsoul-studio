# Remaining Issues (V13.3.1)

Open items, with evidence. Nothing here is marked resolved on the strength of a
unit test alone.

---

## 1. Blockers that prevent a production-ready claim

### 1.1 The 50-portrait unseen matrix cannot be executed here

The specification asks for ≥50 genuinely unseen portraits across multiple
unseen uniform references. This workspace contains:

- **17 portraits** total — 10 in `test_dataset/test_images/`, 7 in
  `test_dataset/extra_images/`
- **4 uniform references**

`test_dataset/README.txt` further states that the T031–T050 samples "are
cropped from the synthetic overview collage and are lower resolution; use real
authorized photos for final validation". Four of the 17 are therefore not
genuine real-world data and are flagged `synthetic_source=True` in the
acceptance matrix, leaving **13 usable real portraits**.

**Missing coverage: 33 of 50 portraits, and all unseen uniform designs.**
Fabricating the remainder, or relabelling previously tested portraits as
unseen, would violate the specification. Not done.

Specifically absent from the available data: single braid, hair bows on a
genuinely unseen subject, tilted heads, off-camera gaze, rotated shoulders,
multiple body proportions, tie/vest/layered uniforms beyond the four
references, and any uniform colour other than red, navy, white and the vest.

### 1.2 Human visual acceptance has not occurred — **PENDING**

The specification requires ≥95% human-verified acceptance. No human review has
been performed. This is marked PENDING and is **not** substituted with
automated gate results.

### 1.3 Fresh inference is 6–7× over the performance target

Measured 366.9–514.3 s per image against a < 60 s target. 3,000 images in 4
hours is not achievable on this hardware at this latency (≈305–373 GPU-hours).
See PERFORMANCE_PROFILING_REPORT.md. Diffusion is 90–95% of wall-clock, so the
remedy is not in the pipeline's non-diffusion stages.

### 1.4 Full fresh-inference matrix not completed

At ~6–8 minutes per case, the full 17-case matrix is 2–2.5 GPU-hours. Only the
cases recorded in `outputs/production_acceptance/provenance_manifest.jsonl`
were executed. **Resumable command:**

```
.venv/Scripts/python.exe scripts/run_production_acceptance.py          # all pending
.venv/Scripts/python.exe scripts/run_production_acceptance.py --list   # status
```

Completed cases are skipped automatically; `--force` re-runs them.

## 2. Known defects still open

### 2.1 Blue halo along the hair crown

Visible in delivered output `PA_11_ShortBoy_Tie` (and in the V13.3 REG_05
artifacts): a thin blue fringe follows the hair boundary, inherited from the
source portrait's blue backdrop. **No gate catches it** — `white_edge_halo`
only looks for white fringes, and background cleanliness passes because the
fringe is inside the subject's alpha, not in the backdrop. This is a real
delivered-quality defect that the 22-gate matrix scores as clean.

### 2.1b Visible rectangular garment seam passes all gates

Delivered output `PA_02_Braids_NavyCollar` (status `completed`, **0 defects**)
contains a clearly visible rectangular boundary across the chest where the
generated placket panel meets the surrounding fabric. The
`rectangular_garment_boundary` gate did not fire, because it looks for straight
*silhouette* edges against the background, not for an internal seam inside the
garment. This is a second delivered-quality defect invisible to the 22-gate
matrix, alongside §2.1.

### 2.1c Asymmetric accessory loss passes the bow-preservation gate

Delivered output `PA_03_MaroonBows_Tie` (status `completed`, **0 defects**).
The source (`maroon_bows.png`) has a maroon ribbon bow on **both** braids. In
the delivery the right-side bow survives and the **left-side bow is gone**,
replaced by a pale blob near the right temple that matches nothing in the
source. `bow_and_ribbon_preservation` passed because it aggregates accessory
pixels rather than requiring each individually-detected accessory to survive
on its own side — so one retained bow masks one erased bow.

This is the third delivered-quality defect that scores 0 (with §2.1 and
§2.1b), and the most serious, because it is a loss of the subject's real
appearance rather than a rendering artefact.

### 2.2 Residual original garment and collar deformation under check fabric

Visible in `PA_12_StraightHair_RedCollar`: the original garment persists at the
lower corners, and the red collar renders as two disconnected lobes rather than
a continuous Peter Pan collar. The case is correctly gated `needs_review`, so
it is not a false delivery — but the gate that fired was the *button* check,
not a collar-shape or residual-garment check. The right defects are not being
named.

### 2.3 Button colour unmeasurable on 2 of 4 references

REF002 and REF004 yield <2 isolable button blobs, so button fidelity returns
UNCERTAIN for them. Better than V13.3's silent pass, but it means the attribute
is unverified for half the available references.

### 2.4 Collar symmetry unmeasurable on same-colour collars

REF003 has a white collar on a white shirt (ΔE 15.8), so the collar cannot be
isolated from the body and symmetry returns UNCERTAIN. A shape- or
edge-based measure, rather than a colour-based one, would be needed.

## 3. Not implemented from the V13.3.1 specification

- **Masked generative repair** and **controlled full garment regeneration**
  (recovery hierarchy levels 3–4), including the configurable retry budget.
- **Defect-specific recovery routes** for button colour/placement, collar
  misregistration, collar colour, collar shape, placket, residual original
  garment, hair-edge contamination and small alpha holes. These are detected
  and gated but not repaired.
- **Iris-based gaze estimation** — explicitly deferred to V13.4 by the
  specification. The current `off_camera_gaze` gate is a nose-to-eye-midpoint
  proxy and must not be cited as gaze verification.
- **Photographic-texture measurements** for the natural-appearance
  requirements (§8). Generative face restoration remains correctly disabled by
  default and protected facial pixels are preserved, but there is no
  quantitative plastic-skin / over-smoothing metric, and ArcFace similarity
  alone is explicitly not sufficient evidence of visual identity preservation.
- **Warm-start vs cold-start separation** in the profiler. Both measured cases
  include model load.
- **Pocket, badge, tie and vest** fidelity still use V13.3 hardcoded thresholds
  (e.g. `mean_torso > 175` for vest presence).

## 4. Pre-existing test failures (not caused by V13.3.1)

`python -m unittest discover -s tests` → 508 tests, 57 failures, 14 errors.
The identical 71 failures occur on the V13.3 baseline (verified by set
comparison before and after: zero new, zero fixed). They belong to the older
`pipelines/qwen_edit_pipeline.py` path. They were not mentioned in any V13.3
report, which claimed "24/24 tests passed".

## 5. Operational guidance (carried forward, still valid)

- Never re-run diffusion solely to change a background colour; recomposite from
  the saved `assembled_foreground_rgba.png`.
- One job in flight: a single 8 GB GPU cannot hold two Qwen jobs.
- Generative face restoration (GFPGAN/CodeFormer) stays disabled by default.
- Cached regression runs (`scripts/run_unseen_regression.py`) are for fast
  development only. **Their latency is cache replay and must never be reported
  as inference performance.**
