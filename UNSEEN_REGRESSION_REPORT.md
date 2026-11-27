# Unseen Regression Report (V13.3.1)

Two distinct bodies of evidence are reported here, kept strictly separate
because conflating them is what made the V13.3 report misleading.

---

## 1. Coverage shortfall — stated first

The specification asks for **≥50 genuinely unseen portraits** across multiple
unseen uniform references. This workspace holds:

| Asset | Count | Note |
| :--- | ---: | :--- |
| Portraits in `test_dataset/test_images/` | 10 | 4 flagged synthetic (below) |
| Portraits in `test_dataset/extra_images/` | 7 | |
| **Usable real portraits** | **13** | |
| Uniform references | 4 | REF001 red collar, REF002 navy collar, REF003 tie, REF004 vest |

`test_dataset/README.txt` states that the T031–T050 samples "are cropped from
the synthetic overview collage and are lower resolution; use real authorized
photos for final validation". Those four cases
(`sunlight`, `blurred`, `scanned`, `complex_background`) are flagged
`synthetic_source=True` in the acceptance matrix and are **not** counted as
unseen real-world coverage.

**Missing: 33 of 50 portraits, and every genuinely unseen uniform design.**
Also absent: single braid, tilted heads, off-camera gaze, rotated shoulders,
varied body proportions, and any collar colour beyond red / navy / white.

No portrait was generated, duplicated or relabelled to close this gap.

## 2. Fresh inference results (verified provenance)

Produced by `scripts/run_production_acceptance.py` with
`raw_diffusion_cache_policy="force_regenerate"`, `diagnostic_mode=False`,
`cached_raw_candidate=None`. Source of record:
`outputs/production_acceptance/provenance_manifest.jsonl`.

| Case | Source | Reference | Status | Provenance | Sim | Latency | Peak VRAM | Defects |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| PA_11_ShortBoy_Tie | blue_bg_boy | REF003_tie | **completed** | verified | 0.9909 | 366.9 s | 7015.6 MB | 0 |
| PA_12_StraightHair_RedCollar | straight_hair | REF001_red_collar | needs_review | verified | 0.9866 | 447.6 s | 7025.6 MB | 1 |
| PA_01_Curly_RedCollar | curly_hair | REF001_red_collar | needs_review | verified (`invocations=1`) | — | 514.3 s | 7014.5 MB | 3 |
| PA_02_Braids_NavyCollar | braids | REF002_navy_collar | **completed** | verified (`invocations=1`) | — | 419.5 s | 7024.5 MB | 0 |
| PA_03_MaroonBows_Tie | maroon_bows | REF003_tie | **completed** | verified (`invocations=1`) | — | 572.6 s | 7034.6 MB | 0 |
| PA_04_NoBows_Vest | no_bows | REF004_vest | needs_review | verified (`invocations=1`) | — | 394.7 s | 7043.4 MB | 1 (`collar_registration_failure`) |
| PA_05_GlassesBindi_RedCollar | glasses_bindi | REF001_red_collar | needs_review | verified (`invocations=1`) | — | 436.4 s | 7051.6 MB | 1 |

**7 cases executed, all with verified fresh-inference provenance.** Latency
min 366.9 s, median 436.4 s, max 572.6 s. Peak VRAM 7014.5–7051.6 MB of 8 GB,
zero CUDA OOM.

Further cases were still executing when this report was written; the manifest
and `--list` give the live status. Nothing is reported here for a case that
did not run.

### 2.1 REG_05 / REG_06 reproduction outcome

- **PA_11 (REG_05 reproduction): resolved.** Under V13.3 this case was gated
  `incomplete_shoulders`. On fresh inference with the corrected gate it
  **delivers with zero defects**. The defect was in the gate, not the image.
- **PA_12 (REG_06 reproduction): correctly gated, wrong reason.** The image has
  genuine defects — a malformed two-lobe collar and residual original garment
  at the lower corners — so blocking delivery is right. But the gate that fired
  was the button check (and that firing was itself a false positive from the
  pre-fix detector; the buttons measure dark navy, matching the reference).
  There is no collar-shape or residual-garment gate naming the real problems.

### 2.2 Caveat on PA_01 and PA_12 defect lists

Both ran before the final button-detector correction, so their recorded
`incorrect_button_color` defects reflect the superseded detector. They are left
in the manifest unaltered rather than edited, and should be re-run. PA_01's
`unwanted_earrings` / `unwanted_accessory` defects are independent of that fix.

## 3. Cache-replay results (NOT inference)

The eight V13.3 `REG_*` rows are retained in `TEST_RESULTS.csv`, relabelled
`Execution Mode = CACHE_REPLAY`, `Inference Invocations = 0`,
`Duration Is Inference Latency = NO`. Those 15–35 s figures measure
compositing, validation and cropping against a stored candidate. Fresh
inference on the same hardware is 367–514 s.

Their quality verdicts are still informative about compositing and gating; only
their timing and their "the pipeline produced this" framing were wrong.

## 4. Comparison artifacts

`scripts/build_comparison_sheets.py` writes a six-panel sheet per executed case
to `outputs/production_acceptance/comparisons/`: source, uniform reference, raw
Qwen candidate, corrected candidate, delivered-or-gated output, and a metrics
panel carrying status, provenance, invocation count, similarity, RMSE, stage
timings, peak VRAM and the full defect list. Rejected candidates are retained
separately under `outputs/production_acceptance/rejected_candidates/`.

## 5. Honest summary

- Executed with verified fresh-inference provenance: **7 of 17** matrix cases.
- Delivered with zero defects **by the gates**: 3 of 7 (PA_11, PA_02, PA_03).
- Delivered and **visually acceptable on inspection: 0 of 3.**
  PA_11 blue hair-crown halo · PA_02 rectangular chest seam · PA_03 one of two
  maroon bows erased. None is detected by any of the 22 gates; all three score
  0 defects.
- Gated for review: 4 of 7 — and in each the *named* defect was imprecise or,
  for the two pre-fix button firings, a false positive from the superseded
  detector.
- False rejections eliminated: 2 (REG_05, REG_06), both verified against the
  original artifacts.
- **Human visual acceptance: PENDING**, and the two findings above are exactly
  why automated gate results cannot substitute for it. A 100% confusion matrix
  coexists with a 0-of-2 visual pass rate on this sample.
