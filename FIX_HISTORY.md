# FIX_HISTORY

Chronological record of defects, root causes and resolutions. Every entry below
was reproduced against a real generated artifact before it was changed, and
re-measured against that same artifact afterwards.

---

## V13.3.1 (2026-11-27)

### FIX-01 — `incomplete_shoulders` false positive on garments near the backdrop colour

**Symptom**: REG_05 (`blue_bg_boy.png` + `REF003_tie.png`) was gated
`needs_review` with `incomplete_shoulders`, blocking delivery of a portrait
whose shoulders are visibly complete.

**Reproduction**: `outputs/unseen_regression/REG_05_ShortBoy_Tie_debug/19_final_quality_validation.json`
→ `{"status": "FAIL", "score": 70, "threshold": 280}`.

**Root cause**: `quality_validator.py` gate 6 measured

```python
shoulder_fg = np.sum(cand_dist_bg[int(h * 0.85), :] > bg_tolerance)
```

`cand_dist_bg` is distance from the *configured background colour*. REG_05 is a
white shirt on a white backdrop. Measured on the delivered RGBA:

| measurement at y = 0.85 h | value |
| :--- | ---: |
| alpha foreground width | 800 / 800 px |
| colour-distance width (tol 30) | 70 px |

The gate was measuring shirt-vs-backdrop contrast and calling it anatomy. It
would fire on any white, cream or pale-grey uniform, and on any garment near
the customer's chosen backdrop colour.

**Fix**: `_measure_shoulder_extent()` measures from the person alpha
(`full_person_alpha`, else the candidate's own alpha channel), intersected with
a **tight** colour tolerance of 4 rather than 30. Erased regions sit exactly on
the backdrop value; real fabric never does (REG_05: 796 px differ by >4, only
70 px by >30). The threshold is now anatomy-relative (1.6 × measured face
width) instead of a fixed 0.35 × canvas width, so narrow children, broad
adults, rotated shoulders and different passport output sizes are judged
against the subject's own build.

**Verification**: REG_05 real artifact 800 ≥ 547 → PASS; alpha-clipped variant
127 < 547 → FAIL; RGB-painted-out variant still FAIL. Gate injection still
detected (44/44). Fresh re-run PA_11 delivered with **0 defects**.

---

### FIX-02 — `collar_registration_failure` false positive from sleeve trim

**Symptom**: REG_06 (`straight_hair.png` + `REF001_red_collar.png`) gated with
`Collar center offset horizontally from facial axis (offset=0.339 > 0.16)`,
although the collar is correctly placed.

**Root cause**: the check took the centroid of *every* red pixel in a window
±0.50 face widths wide. REF001 has a red Peter Pan collar **and red sleeve
cuffs**. The red histogram across that window was bimodal:

```
x:  226  254  283  311  340  368  397  425  454  482  511  539
n:   41   57    0    0   27  111  188   44    0  162 1002  993
         ^ collar (near the face axis)              ^ sleeve cuff
```

76% of the red was sleeve trim. A centroid over two separated clusters lands
between them, far from the axis. The check also only ran for red collars and
reported facial-landmark RMSE as its score.

**Fix**: `_measure_collar_symmetry()` compares left- and right-side collar
attachment about the facial axis, inside the neck corridor only
(y ∈ [chin − 0.20 fh, chin + 0.25 fh], x within ±0.55 fw, which excludes cuffs
and shoulder piping). It is colour-agnostic: the collar colour comes from the
reference analyzer's measured CIELAB value, so pointed, rounded, Peter Pan,
mandarin, polo and open collars in any colour are handled identically. When the
reference collar is not separable from its own body fabric it returns
**UNCERTAIN**, never an unmeasured pass.

**Verification**: REG_06 real artifact imbalance 0.048 → PASS; one-sided collar
1.000 → FAIL; green collar → PASS; collar-equals-body → UNCERTAIN. The
collar-displacement injection is still caught by the throat-length branch.

---

### FIX-03 — `incorrect_button_color` measured fabric, not buttons

**Symptom**: REG_06 gated `incorrect_button_color (white buttons generated
instead of black)`.

**Root cause**: the check counted *pixels* brighter than 195 inside a 60 px
strip and failed when >1500 of them appeared with <20 dark pixels. Measured on
REG_06 with the real detected face box: **dark = 1, white = 4856** — every one
of those 4856 pixels a white square of REF001's gingham check. The threshold
was a property of the fabric, not of the buttons. The branch also only existed
for `expected_btn == "black"`; white, gold or fabric-matched buttons were never
checked and silently passed.

**Fix**: buttons are detected as **blobs** by `measure_button_blobs()`, shared
by the reference analyzer and the candidate validator so both sides measure
identically: components thresholded on CIELAB deviation from the placket's own
median cloth colour, filtered by size, roundness, solidity and proximity to the
placket centreline, sampled at the blob **interior** (the rim blends button and
cloth), then the colour the largest blobs agree on. Comparison is ΔE76 against
the reference's measured button colour, with **UNCERTAIN** when fewer than two
buttons can be isolated. No colour is hardcoded.

Two intermediate implementations were rejected by measurement before this one:
a plain dominant-colour estimate returned the gingham (REF001 → RGB 188,196,216
instead of near-black), and Otsu over a local-contrast image returned the
fabric again on PA_12 (reported *white* buttons for a placket whose buttons are
plainly dark navy, RGB 46,54,67).

**Verification**: REF001 reference buttons measure RGB (27,29,33) ≈ black, which
matches the visible garment. PA_12's candidate buttons measure (46,54,67) and
correctly compare as a match. White-on-mid-tone vs black reference → FAIL; gold
vs gold → PASS; no buttons → UNCERTAIN; black buttons on gingham → measured
dark, not bright.

---

### FIX-04 — Recovery could ship a defect the candidate never had

**Root cause**: `RecoveryController` accepted any attempt whose defect **count**
decreased. An attempt removing two defects while introducing a third was
therefore accepted, shipping a new defect.

**Fix**: any attempt that introduces a defect type absent from the pre-repair
set is rejected and rolled back, compared on `_defect_key()` so the same defect
at a different measurement is not mistaken for a new one.

---

### FIX-05 — Shoulder "reconstruction" was rectangular expansion

**Root cause**: `_repair_incomplete_shoulders` pushed every torso row out to
fixed canvas fractions (0.12 w / 0.88 w) — the naive expansion the V13.3.1
specification prohibits. It squares the silhouette, overrides the subject's
build, and on a rotated portrait widens each side by a different wrong amount.

**Fix**: the outer shoulder contour is convex — it widens as it descends, then
flattens, and never narrows and widens again. The repair computes the **lower
convex hull** of each side's edge profile; rows bulging inside that hull are
the clipped stretch, and the hull supplies the slope to restore them to.
Gross one-sided loss (reach asymmetry > 0.45 about the face axis) and
deficits > 18% of torso area are **declined** and routed to review rather than
inpainted into visible distortion.

A running-extreme envelope and a straight-line fit were both tried and rejected
by measurement: for a monotonically widening edge the running extreme always
equals the current value and detects nothing, and a line fitted through a
contour that plateaus predicts an ever-widening body and "repairs" an intact
silhouette. Verified against ground truth: hull 75.0 vs true 75, 58.8 vs 59.

---

### FIX-06 — Uniform fidelity hardcoded to red and navy

**Root cause**: `fidelity_validator.py` branched on the literal strings `"red"`
and `"navy blue"` with hand-tuned RGB/HSV windows. Every other collar colour
fell through to `{"status": "PASS", "score": 1, "threshold": 1}` — an
unmeasured pass. This is why "universal uniform support" could be asserted
while only two colours were ever verified.

**Fix**: CIELAB ΔE76 comparison against the reference's measured collar colour
(tolerance 28, which absorbs fabric shading and white balance while separating
one garment colour from another), falling back to the legacy branches only for
specs predating `collar_lab`, and returning **UNCERTAIN** when nothing can be
measured. Button fidelity routes through the same shared blob detector.

---

### FIX-07 — Regression runner could not prove fresh inference

**Root cause**: `scripts/run_unseen_regression.py` runs with
`raw_diffusion_cache_policy="authoritative_cache"`, `diagnostic_mode=True` and
an explicit `cached_raw_candidate`, so it replays stored candidates and never
invokes Qwen. Its 15–35 s timings were published as pipeline performance.
`scripts/benchmark_fresh_inference.py` has a related defect: when a prior trace
exists it parses that instead of running, yet still writes
`Fresh Inference: True` together with **hardcoded** VRAM constants
(`peak_alloc = 5824.0`), which is why every row of the old
`FRESH_INFERENCE_BENCHMARK.csv` shares identical VRAM figures.

**Fix**: new `scripts/run_production_acceptance.py` pins
`force_regenerate` / `diagnostic_mode=False` / `cached_raw_candidate=None`,
counts real generator invocations by patching `UniformGenerator.generate` on the
**class** (the orchestrator binds the singleton at import, so patching the
module attribute would never be seen), checks the raw candidate file was
rewritten during the run, records SHA-256 hashes and a unique execution id per
case, writes a JSONL provenance manifest, and is resumable.
