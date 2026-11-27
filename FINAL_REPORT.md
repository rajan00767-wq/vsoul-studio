# V13.3.1 Final Report — Production Verification, Uniform Fidelity & Defect Recovery

**Status: NOT PRODUCTION READY**
**Date**: 2026-11-27
**Hardware**: NVIDIA RTX 5050 Laptop GPU, 8 GB VRAM, Windows 11
**Pipeline**: Qwen-Image-Edit-2511 Q4_K_M GGUF + 4-step Lightning LoRA

Status selected strictly on verified evidence. The pipeline is, on current
evidence, suitable for **supervised pilot use with mandatory human review of
every delivery**, but it cannot be called production ready: no human visual
acceptance has been performed, the unseen evaluation matrix cannot be
assembled from the data in this workspace, the performance target is missed by
6–7×, and at least one delivered-quality defect passes all 22 gates.

---

## 1. Executive summary

V13.3.1 was asked to verify V13.3's claims rather than trust them. Verifying
them against the actual code and the actual generated artifacts found that
**several central claims were false**, and that two of the three "authentic
generative defects" V13.3 reported were in fact **bugs in the quality gates**
that were rejecting correct portraits.

| V13.3 claim | Verdict | Finding |
| :--- | :--- | :--- |
| "REG_05 contained an authentic `incomplete_shoulders` defect" | **FALSE** | Shoulders span the full 800 px canvas. The gate measured *distance from the backdrop colour*, and the shirt was white on a white backdrop: 800 px of alpha, 70 px of colour difference. Any pale uniform triggered it. |
| "REG_06 … `collar_registration_failure`" | **FALSE** | The collar is correctly placed. The check took a centroid over all red pixels in a wide window; 76% of that red was REF001's red **sleeve cuffs**, pulling the centroid off-axis. |
| "REG_06 … `incorrect_button_color`" | **FALSE as stated** | The gate counted pixels brighter than 195 in a 60 px strip — 4856 of them, every one a white square of the **gingham check fabric**, with zero buttons involved. |
| "Zero false deliveries" | **True but vacuous** | Achieved partly through false *rejections*. 2 of 8 gated cases were gate bugs. |
| "Unseen regression: 15.6–35.1 s per case" | **FALSE as presented** | That suite replays cached candidates and never invokes Qwen. Verified fresh inference is **367–514 s**. |
| "Dynamic reference analysis, zero hardcoded templates" | **FALSE** | The validators branched on the literal strings `"red"` and `"navy blue"`; every other collar colour hit an `else` that returned PASS **without measuring anything**. |
| "24/24 tests passed, 100% across all modules" | **Misleading** | True of 7 V13 modules. The full suite is 490 tests with **71 pre-existing failures** no V13.3 report mentions. |
| "`Fresh Inference: True`" in the V13.3 benchmark CSV | **FALSE** | That script re-parses a prior trace when one exists, yet still writes `True` alongside **hardcoded** VRAM constants — which is why rows share identical memory figures. |

## 2. What was fixed

Seven defects, each reproduced against a real artifact before the change and
re-measured against it afterwards. Full detail in **FIX_HISTORY.md**.

1. **Shoulder gate** — now measures person/candidate alpha intersected with a
   tight (tol 4) colour tolerance, with an anatomy-relative width requirement
   (1.6 × measured face width) rather than a fixed canvas fraction. Handles
   narrow children, broad adults, rotated shoulders and any output size.
2. **Collar registration** — colour-agnostic left/right symmetry about the
   facial axis inside the neck corridor, excluding sleeve zones. Works for
   pointed, rounded, Peter Pan, mandarin, polo and open collars in any colour.
3. **Button fidelity** — buttons detected as blobs (CIELAB deviation from the
   placket's own cloth colour, filtered by size/roundness/solidity/centreline,
   sampled at the blob interior), compared by ΔE76 against the reference's
   measured button colour. No colour is forced.
4. **Uniform fidelity** — CIELAB comparison for arbitrary collar colours,
   replacing the red/navy branches and the unmeasured `else` pass.
5. **Recovery rollback** — an attempt introducing a defect type the candidate
   did not have is rejected and rolled back, instead of being accepted because
   the defect *count* fell.
6. **Shoulder reconstruction** — lower-convex-hull contour reconstruction
   replacing fixed-canvas rectangular expansion, with explicit declines for
   gross one-sided loss (>0.45 reach asymmetry) and large deficits (>18% torso).
7. **Fresh-inference enforcement** — new `scripts/run_production_acceptance.py`
   pins `force_regenerate` / `diagnostic_mode=False` / no cached candidate,
   counts real generator invocations, verifies candidate rewrite, and records
   hashes, execution ids and timestamps in a resumable JSONL manifest.

**UNCERTAIN** was introduced throughout: an attribute that cannot be reliably
measured is now reported as uncertain, never as a silent pass.

## 3. Verified results

| Metric | Result |
| :--- | :--- |
| 22 mandatory gates (44 conditions) | TP 22, TN 22, FP 0, FN 0 — 100% |
| 12 studio quality gates | 12/12 |
| V13 + V13.3.1 module tests | 47/47 |
| New failures introduced | **0** (failing set identical before and after: 71 pre-existing) |
| Fresh-inference cases executed | 7 of 17, all forced-regeneration |
| Provenance | 5 `fresh_inference_verified` (`invocations=1`), 2 verified by candidate-rewrite |
| Peak VRAM | 7014.5–7051.6 MB of 8 GB, zero OOM |
| Fresh latency | 366.9–572.6 s/image, median 436.4 s (diffusion = 90–95% of wall-clock) |

Two regressions were caught *during* development by running the gate suite
rather than assuming — a button rewrite that produced 21 false positives, and
an alpha-only shoulder measure that missed the injection. Both are recorded in
QUALITY_GATE_REGRESSION_REPORT.md §3 rather than hidden behind a final green
result.

## 4. Why the status is NOT PRODUCTION READY

- **No human visual acceptance** has been performed. The spec requires ≥95%;
  this is PENDING, not substituted with automated results.
- **The unseen matrix cannot be built here.** 17 portraits exist (4 flagged
  synthetic by `test_dataset/README.txt` itself) and 4 uniform references.
  33 portraits and all genuinely unseen uniform designs are missing.
  Fabricating them, or relabelling tested portraits as unseen, was refused.
- **Performance target missed 6–10×.** Median 436.4 s against < 60 s; 3,000
  images would need ≈364 GPU-hours, 91× the 4-hour budget.
- **Every delivered image carries an undetected defect — 3 of 3.**
  PA_11 a blue halo along the hair crown (`white_edge_halo` only looks for
  *white* fringes); PA_02 a visible rectangular seam across the chest
  (`rectangular_garment_boundary` checks the silhouette, not seams inside the
  garment); PA_03 **one of the subject's two maroon bows erased** while the
  other survives (`bow_and_ribbon_preservation` aggregates accessory pixels, so
  a retained bow masks an erased one). The 22-gate matrix scores all three
  **clean, 0 defects**.

  This is the single most important finding of V13.3.1: the gate suite scores
  100% while the delivered images have a **0-of-3 visual pass rate**. A perfect
  confusion matrix measures the injections, not the product. Human review
  cannot be skipped, and PA_03 shows the gates can miss outright loss of the
  subject's real appearance, not merely cosmetic artefacts.
- **Universality is unproven.** The fidelity code path is colour-agnostic and
  tested on synthetic arbitrary colours, which proves the path, not the
  product. No claim of universal uniform compatibility is made.

## 5. Deliverables

`V13_3_1_IMPLEMENTATION_PLAN.md`, `FINAL_REPORT.md`, `FIX_HISTORY.md`,
`REMAINING_ISSUES.md`, `QUALITY_GATE_REGRESSION_REPORT.md`,
`UNIFORM_FIDELITY_REPORT.md`, `RECOVERY_EFFECTIVENESS_REPORT.md`,
`STUDIO_POSE_VERIFICATION_REPORT.md`, `PERFORMANCE_PROFILING_REPORT.md`,
`PRODUCTION_READINESS_CHECKLIST.md`, `FRESH_INFERENCE_BENCHMARK.csv`,
`TEST_RESULTS.csv`, plus generated outputs, rejected candidates and the
provenance manifest under `outputs/production_acceptance/`.

## 6. Resuming the evaluation

```bash
.venv/Scripts/python.exe scripts/run_production_acceptance.py --list   # status
.venv/Scripts/python.exe scripts/run_production_acceptance.py          # all pending
```

Completed cases are skipped via the manifest; budget 6–8 minutes per case.
