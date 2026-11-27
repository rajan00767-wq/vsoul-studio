# Quality Gate Regression Report (V13.3.1)

> **Maintenance note**: `scripts/verify_quality_gates.py` **overwrites this
> file** with a short auto-generated summary every time it runs. This document
> is the authored V13.3.1 analysis. If it has reverted to a
> "V13.2 Quality Gate Regression & Independent Verification Report" heading,
> the script regenerated it and this version should be restored from git. The
> raw confusion matrix is identical either way; only the analysis is lost.

Source of all figures: `scripts/verify_quality_gates.py`, executed against the
V13.3.1 working tree on 2026-11-27.

---

## 1. Confusion matrix — 22 gates, 44 conditions

```
                      Defect Condition (Actual Positive)   Clean Baseline (Actual Negative)
Test Result (Positive)            TP = 22                              FP = 0
Test Result (Negative)            FN = 0                               TN = 22

Precision            : 100.00%
Recall (Sensitivity) : 100.00%
Specificity          : 100.00%
Overall Accuracy     : 100.00%
Delivery Rejection   : 100.00% (22/22 defective candidates blocked)
```

Identical to the V13.3 figures, which is the point: the three gate rewrites in
V13.3.1 removed false positives on **real** portraits without weakening
detection of **injected** defects.

## 2. What this matrix does and does not prove

It proves the gates behave correctly on one clean baseline portrait and 22
programmatic defect injections. It proves very little about real generative
output, and V13.3.1 produced direct evidence of that in both directions:

**False positives this matrix never saw.** REG_05 and REG_06 both scored 100%
here while falsely rejecting correct portraits, because the injections
differed from a real white shirt or a real red-trimmed uniform in exactly the
ways the gates keyed on.

**False negatives this matrix never saw.** All three images delivered during
fresh-inference acceptance scored **0 defects** and all three carry a visible
defect:

| Case | Defect | Why no gate fired |
| :--- | :--- | :--- |
| PA_11 | Blue halo along the hair crown | `white_edge_halo` only looks for *white* fringes |
| PA_02 | Rectangular seam across the chest | `rectangular_garment_boundary` checks the silhouette, not seams inside the garment |
| PA_03 | One of two maroon bows erased | `bow_and_ribbon_preservation` aggregates accessory pixels, so a surviving bow masks an erased one |

**A 100% confusion matrix coexists with a 0-of-3 visual pass rate.** This
report must therefore always be read next to `UNSEEN_REGRESSION_REPORT.md`,
never alone, and never cited as evidence of delivery quality.

## 3. Regressions caught during V13.3.1 development

The rewrites were caught breaking this suite twice, by running it rather than
assuming:

| Stage | TP | TN | FP | FN | Cause |
| :--- | --: | --: | --: | --: | :--- |
| V13.3 baseline (changes stashed) | 22 | 22 | 0 | 0 | — |
| After first button rewrite | 21 | 1 | **21** | 1 | Blob colour sampled at the component rim, which blends button and cloth; the clean baseline's black buttons measured (119,109,134), failing every clean case. |
| After interior sampling | 22 | 22 | 0 | **1** | Alpha-only shoulder measurement missed the injection, which erases RGB but leaves person alpha intact. |
| **V13.3.1 final** | **22** | **22** | **0** | **0** | Alpha ∧ tight (tol 4) colour distance. |

The middle rows are recorded deliberately: both were real regressions that a
"tests pass" summary would have concealed.

## 4. Gates changed in V13.3.1

All 22 gates: negative injection FAIL, clean baseline PASS, delivery blocked.
Unchanged from V13.3 except these three.

| # | Gate | V13.3.1 change |
| --: | :--- | :--- |
| 10 | `incomplete_shoulders` | Measures person/candidate alpha ∧ tight colour tolerance, with an anatomy-relative width requirement (1.6 × face width) replacing 0.35 × canvas width. |
| 17 | `collar_registration_failure` | Throat-length branch unchanged. Red-centroid branch replaced by colour-agnostic left/right symmetry inside the neck corridor, excluding sleeve zones. |
| — | `button_color` | Blob-based, any colour, ΔE76 against the reference's measured button colour; UNCERTAIN when unmeasurable. |

## 5. Studio quality gates

`tests/test_studio_quality_gates.py`: **12/12 pass**, unchanged by V13.3.1.

Gate 4 (`off_camera_gaze`) measures a nose-to-eye-midpoint proxy and is **not**
an authoritative gaze measurement — see STUDIO_POSE_VERIFICATION_REPORT.md.
Its passing does not mean gaze has been verified.

## 6. Unit test suite

| Suite | Result |
| :--- | :--- |
| V13 module suites + V13.3.1 suites (10 modules) | **47 / 47 pass** |
| Full `unittest discover -s tests` | 513 tests · 57 failures · 14 errors · 1 skipped |

The full-suite failures are **pre-existing and unrelated**: the identical 71
failures occur on the V13.3 baseline before any V13.3.1 change, verified by set
comparison run both before and after the work (zero new, zero fixed). They
belong to the older `pipelines/qwen_edit_pipeline.py` path, not
`pipelines/uniform_swap/`. V13.3's reports claimed "24/24 tests passed, 100%
pass across all 5 test modules", which was true only of the V13 subset and
omitted the other ~466 tests.

## 7. Recommended gate additions (not implemented)

Directly implied by §2's false negatives:

1. **Coloured edge halo** — generalize `white_edge_halo` to any fringe colour
   that does not occur in the source along that boundary.
2. **Internal garment seam** — detect straight high-contrast edges *inside* the
   garment mask, not only on the silhouette.
3. **Per-accessory survival** — require each individually detected source
   accessory to survive on its own side, instead of aggregating pixel counts.
