# V13.3.1 Implementation Plan — Verified Audit of V13.3

**Audit date**: 2026-11-27
**Method**: every claim below was verified against actual code and actual generated artifacts in `outputs/unseen_regression/`. Claims in `FINAL_REPORT.md`, `REMAINING_ISSUES.md` and `UNSEEN_REGRESSION_REPORT.md` were **not** taken at face value; where they disagree with the artifacts, the artifacts win.

---

## 0. Audit summary

| Prior claim (V13.3 reports) | Verified? | Evidence |
| :--- | :--- | :--- |
| "44/44 quality gate tests pass" | **Partly** | `tests.test_studio_quality_gates`, `test_v13_3_components`, `test_v12_4_quality_validation`, `test_modular_uniform_swap`, `test_force_regenerate_policy`, `test_passport_and_multibackground`, `test_pose_analyzer` = **24 tests, all pass**. But the *full* suite is 490 tests with **71 pre-existing failures** (57 F / 14 E), none of which any V13.3 report mentions. |
| "Unit Test Suite: 24/24 (100% pass across all 5 modules)" | **Misleading** | True only for the 7 V13 modules. `python -m unittest discover -s tests` → `Ran 490 tests … FAILED (failures=57, errors=14, skipped=1)`. |
| "REG_05 contained an authentic generative defect (`incomplete_shoulders`)" | **FALSE** | The raw Qwen candidate has full, correct shoulders (verified visually and by pixel measurement). The gate is a **false positive**. See §2. |
| "REG_06 … `incorrect_button_color`" | **FALSE as stated** | The reference's buttons *are* black and the candidate's placket genuinely lacks them, but the gate that fires measures the wrong thing and would fire on most light-fabric uniforms. See §3. |
| "REG_06 … `collar_registration_failure`" | **FALSE positive** | The collar is correctly registered; the check's red-pixel centroid is dragged right by the red **sleeve cuffs**, which are inside its sampling window. See §4. |
| "Zero false deliveries" | **True but vacuous** | Zero false deliveries was achieved partly by false *rejections*. 2/8 gated cases are gate bugs, not generation defects. |
| "Unseen regression … 35.1s / 15.6s / 22.5s per case" | **FALSE as presented** | `scripts/run_unseen_regression.py:201-219` sets `raw_diffusion_cache_policy="authoritative_cache"`, `diagnostic_mode=True` and passes `cached_raw_candidate=outputs/{tid}_qwen_raw.png`. **No Qwen inference ran.** Those are compositing-only latencies. `FRESH_INFERENCE_BENCHMARK.csv` shows real fresh latency is **375–870 s**. |
| "Dynamic reference analysis, zero hardcoded templates" | **FALSE** | `fidelity_validator.py` branches on literal `"red"` / `"navy blue"` collar strings and hardcoded HSV/RGB thresholds; any other collar colour silently returns PASS without measuring anything. See §5. |
| "Production target < 60 s/image, 3000 images in 4 h" | **Not achieved, not claimed as achieved in code** | Measured fresh inference is 375–870 s. 3000 images at 375 s ≈ 312 GPU-hours. |

---

## 1. Confirmed defects, root causes, fixes, risks

### §2 — DEFECT 1: `incomplete_shoulders` false positive (REG_05)

**Reproduced.** `outputs/unseen_regression/REG_05_ShortBoy_Tie_debug/19_final_quality_validation.json`:
`incomplete_shoulders {status: FAIL, score: 70, threshold: 280}`.

**Root cause** — `pipelines/uniform_swap/quality_validator.py`, gate 6:

```python
y_shoulder = int(h * 0.85)
shoulder_fg = int(np.sum(cand_dist_bg[y_shoulder, :] > bg_tolerance))
```

`cand_dist_bg` is *colour distance from the configured background colour*. REG_05's background is white **and the uniform is a white shirt**. Measured on the actual artifact:

```
17_assembled_foreground_rgba.png @ y=0.85h
  alpha foreground width  = 800 / 800 px   (shoulders are complete)
  colour-distance width   =  70 / 800 px   (white shirt reads as background)
```

The subject's shoulders span the full canvas. The gate counts 70 px because only the navy tie and a little shading differ from white. This fires on **every white or near-background-coloured garment**, which is a large fraction of school uniforms.

**Fix**: measure shoulder extent from the **alpha / person mask** (`identity_layer["full_person_alpha"]`, or the RGBA alpha channel) and fall back to colour distance only when no alpha is available. Gate 7 (`foreground_alpha_hole`) already does exactly this — the fix makes gate 6 consistent with it.

**Also**: replace the fixed `y = 0.85h`, `min_width = 0.35w` constants with anatomy-relative measurement (shoulder row derived from chin_y + face_height, expected width derived from measured face width), so narrow children and broad adults, rotated shoulders and different passport output sizes are all handled without a magic canvas fraction.

**Regression risk**: `tests/test_studio_quality_gates.py` and `scripts/verify_quality_gates.py` inject this defect by narrowing the subject; the injection is alpha-visible as well as colour-visible, so detection must be preserved. New test must assert *both* directions (white shirt on white bg → PASS; genuinely clipped shoulders → FAIL).

### §3 — DEFECT 2: `incorrect_button_color` (REG_06)

**Reproduced.** Gate at `quality_validator.py:~617-633`:

```python
placket_box = cand_rgb[chin+0.15fh : chin+0.85fh, mid_x-30 : mid_x+30]
dark_button_px  = count(all channels < 55)
white_button_px = count(all channels > 195)
if dark_button_px < 20 and white_button_px > 1500: FAIL
```

Measured on the actual REG_06 output with the real detected face box
(`bbox [226.6, 310.6, 569.0, 734.5]`): placket zone y 797–1024, **dark = 1 px, white = 4856 px** → FAIL.

**Root causes, two of them:**

1. **Wrong measurement.** This counts *fabric* pixels, not buttons. REF001's fabric is a white/light-blue gingham check; a 60 px-wide strip of it contains thousands of >195 "white" pixels regardless of button colour. The threshold `white > 1500` is a property of the fabric, not of the buttons.
2. **Hardcoded polarity.** The branch only exists for `expected_btn == "black"`. A uniform with white, gold, or fabric-matched buttons is never checked at all — it silently returns `PASS`.

The underlying *visual* issue in REG_06 is real (the candidate's placket has no visible buttons), but this gate does not measure it and would misfire on a correct candidate.

**Fix**: detect buttons as **blobs**, not pixels — a vertical run of small, compact, locally-contrasting connected components along the placket centreline. Compare the measured blob colour (CIELAB) against the reference's measured button colour, with an explicit **UNCERTAIN** result when too few blobs are found to decide. Never force black.

**Regression risk**: `verify_quality_gates.py` injects white buttons on a black-button spec. The injection must still be caught via blob colour, so the injected defect needs to be a real blob, not a painted rectangle — verify and, if necessary, strengthen the injection.

### §4 — DEFECT 3: `collar_registration_failure` false positive (REG_06)

**Reproduced.** `quality_validator.py:~1407-1422`:

```python
neck_zone = cand_rgb[chin_y : chin_y+0.40fh, fcx-0.50fw : fcx+0.50fw]
is_red_col = red-threshold mask
col_cx = mean(x of red pixels)
if |col_cx - fcx| / fw > 0.16:  FAIL
```

Measured on the real artifact: `col_cx = 507.2`, `fcx = 397.8`, `offset = 0.320` → FAIL.

**Root cause**: the red-pixel histogram across the sampling window is

```
x:  226  254  283  311  340  368  397  425  454  482  511  539
n:   41   57    0    0   27  111  188   44    0  162 1002  993
```

Two clusters. The left/central cluster is the **collar**. The 511/539 cluster (≈2000 px, 76% of all red in the window) is the **red sleeve cuff/trim** of REF001 entering the window on the viewer's right. A centroid over both clusters lands between them, far from the face axis. Nothing is wrong with the collar.

Secondary flaws in the same gate: it only runs for `has_red_collar`; it uses a centroid instead of a symmetry measure; and it re-uses `geometry_rmse` (a *facial landmark* metric) as the collar's reported score, which is misleading in the JSON.

**Fix**: restrict collar measurement to the collar region (bounded to the neck corridor, excluding lateral sleeve zones), measure **left/right collar attachment symmetry** about the facial axis instead of a global centroid, and make it colour-agnostic — operate on whatever collar colour the reference analyzer measured, with an UNCERTAIN result when the collar cannot be isolated.

**Regression risk**: `collar_registration_failure` injection displaces the collar down by `0.49*fh`; the throat-length branch (unchanged) catches that. Verify it still does.

### §5 — DEFECT 4: hardcoded uniform fidelity thresholds

`pipelines/uniform_swap/fidelity_validator.py` branches on `collar_color == "red"` / `"navy blue"` with literal RGB/HSV windows, on `expected_btn == "black"`, on `mean_torso > 175` for vests, and otherwise writes `{"status": "PASS", "score": 1, "threshold": 1}` — i.e. **an unmeasured pass**. This directly contradicts "zero hardcoded templates".

**Fix**: foreground-aware dominant-colour extraction + CIELAB ΔE comparison between reference ROI and candidate ROI, for arbitrary colours; explicit `UNCERTAIN` status instead of an unmeasured `PASS`.

### §6 — DEFECT 5: regression runner does not prove fresh inference

`scripts/run_unseen_regression.py` uses the cache (see §0). Fix: a separate `scripts/run_production_acceptance.py` that pins `raw_diffusion_cache_policy="force_regenerate"`, `diagnostic_mode=False`, `cached_raw_candidate=None`, counts actual Qwen invocations, records hashes and timestamps, and is resumable. The cached runner stays for fast dev but must label its latency as cache-replay.

---

## 2. Affected modules

| Module | Change |
| :--- | :--- |
| `pipelines/uniform_swap/quality_validator.py` | shoulder gate → alpha-based + anatomy-relative; button gate → blob-based, colour-agnostic; collar registration → symmetry-based, sleeve-excluded, colour-agnostic |
| `pipelines/uniform_swap/fidelity_validator.py` | CIELAB reference-driven comparison, UNCERTAIN status |
| `pipelines/uniform_swap/recovery_controller.py` | defect-specific routes, rollback, post-recovery re-gating |
| `scripts/run_production_acceptance.py` (new) | fresh-inference enforcement + provenance + resume |
| `scripts/run_unseen_regression.py` | label cache-replay latency honestly |
| `tests/` | new tests per §11 of the brief |

## 3. Known blockers (stated up front, not discovered later)

- **50-portrait unseen matrix is not executable in this environment.** `test_dataset/` contains **17** portraits total (10 in `test_images/`, 7 in `extra_images/`) and **4** uniform references. `test_dataset/README.txt` itself states T031–T050 are *cropped from a synthetic collage*, i.e. not genuine unseen real-world data. Fabricating the remaining 33 would violate the brief. Missing coverage will be reported explicitly.
- **Fresh inference cost.** Measured 375–870 s/image on the RTX 5050. A 50-case fresh matrix is 5–12 GPU-hours. The acceptance runner is therefore built resumable; what actually executes will be reported, and nothing else.
- **Human visual acceptance cannot be performed by this process.** It will be marked PENDING, never claimed.
- **71 pre-existing full-suite test failures** are outside V13.3.1's scope (they belong to the older `pipelines/qwen_edit_pipeline.py` path, not `pipelines/uniform_swap/`). They are recorded as a baseline so that no new failure is hidden among them.
