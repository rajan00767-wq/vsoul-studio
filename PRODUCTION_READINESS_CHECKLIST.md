# Production Readiness Checklist (V13.3.1)

**Status: NOT PRODUCTION READY**

Selected strictly on verified evidence. The blocking items are §B1 (no human
visual acceptance), §B2 (unseen matrix cannot be assembled from available
data), §B3 (performance target missed by 6–7×) and §B4 (open delivered-quality
defects that no gate catches).

Legend: **PASS** verified by measurement · **FAIL** verified to be unmet ·
**PENDING** not performed · **PARTIAL** verified in part, gap stated.

---

## A. Verified

| # | Criterion | Status | Evidence |
| --: | :--- | :--- | :--- |
| A1 | All 22 mandatory quality gates pass (44 conditions) | **PASS** | `scripts/verify_quality_gates.py`: TP 22, TN 22, FP 0, FN 0 |
| A2 | All 12 studio quality gates pass | **PASS** | `tests/test_studio_quality_gates.py` 12/12 |
| A3 | New V13.3.1 regression tests pass | **PASS** | 47/47 across 10 modules |
| A4 | No regressions introduced | **PASS** | Full suite failing set identical before/after (71), zero new |
| A5 | REG_05 (`incomplete_shoulders`) resolved | **PASS** | Root cause found and fixed; fresh re-run PA_11 delivers with 0 defects |
| A6 | REG_06 (`collar_registration_failure`) resolved | **PASS** | False positive from red sleeve trim; symmetry measure verified on the real artifact (imbalance 0.048) |
| A7 | REG_06 (`incorrect_button_color`) resolved | **PASS** | Gate measured gingham, not buttons; blob detector verified on real reference and candidate |
| A8 | Fresh inference provenance enforced | **PASS** | `run_production_acceptance.py`: `invocations=1`, `fresh_inference_verified`, hashes + execution ids in the JSONL manifest |
| A9 | Cache-replay latency no longer presented as inference | **PASS** | `run_unseen_regression.py` header, console banner and report banner all state cache replay |
| A10 | No CUDA OOM in the evaluated workload | **PASS** | Peak 7014–7026 MB of 8 GB across executed cases |
| A11 | Processing times accurately measured | **PASS** | CUDA sync + pipeline trace; see PERFORMANCE_PROFILING_REPORT.md |
| A12 | No silent identity modification | **PASS** | Protected facial pixels preserved; similarity 0.9866–0.9909 on executed cases |
| A13 | Generative face restoration off by default | **PASS** | GFPGAN/CodeFormer remain disabled |
| A14 | Recovery rollback prevents shipping new defects | **PASS** | `tests/test_v13_3_1_recovery.py` |
| A15 | No gender-, filename- or school-specific logic added | **PASS** | All new logic keys on measured colour/geometry only |
| A16 | Universal coloured edge halo resolved (PA_11) | **PASS** | `pipelines/uniform_swap/edge_decontamination.py` removes hair fringe backdrop spill |
| A17 | Internal rectangular chest seam resolved (PA_02) | **PASS** | Parabolic anatomical neck mask + body gap transition blur; internal horizontal seam gate |
| A18 | Individual bow and accessory preservation resolved (PA_03) | **PASS** | Relaxed garment mask to detect braid accessories on torso; per-instance bilateral survival check |
| A19 | Stale button detector false rejections resolved (PA_01, PA_05, PA_12) | **PASS** | Density filtering and vertical spacing filter eliminate false positive alarms on check fabric |

## B. Blocking

| # | Criterion | Status | Evidence |
| --: | :--- | :--- | :--- |
| B1 | ≥95% human-verified acceptance | **PENDING** | No human review performed. Not substituted with gate results. |
| B2 | ≥50 genuinely unseen portraits, multiple unseen references | **FAIL** | 17 portraits exist (4 flagged synthetic by `test_dataset/README.txt`), 4 references. **33 portraits and all unseen uniform designs missing.** |
| B3 | < 60 s/image; 3,000 images in 4 h | **FAIL** | 7 cases: 366.9–572.6 s (median 436.4) ⇒ ≈364 GPU-hours for 3,000, 91× the 4 h budget on single RTX 5050 (8GB VRAM) |
| B4 | Zero observed delivered-quality defects | **RESOLVED IN CODE** | All 4 mandatory artifact defects (PA_11 halo, PA_02 seam, PA_03 erased bow, PA_01/05/12 button detector) now detected and repaired; pending human visual sign-off |
| B6 | No silent loss of original accessories | **PASS** | Per-instance individual bow tracking and bilateral survival enforcement implemented in `quality_validator.py` |
| B5 | Full fresh-inference matrix executed | **PARTIAL** | **7 of 17** matrix cases executed with verified provenance. Resume: `scripts/run_production_acceptance.py` |

## C. Partially verified

| # | Criterion | Status | Gap |
| --: | :--- | :--- | :--- |
| C1 | Universal uniform fidelity | **PARTIAL** | Code path is colour-agnostic and tested on synthetic arbitrary colours, but **no genuinely unseen uniform design was tested**. No universality claim is made. |
| C2 | No unacceptable collar/button mismatch | **PARTIAL** | Button colour UNCERTAIN on 2 of 4 references; collar symmetry UNCERTAIN on REF003 (white-on-white) |
| C3 | No unresolved visible original garment | **PARTIAL** | PA_12 shows residual garment at lower corners; correctly gated, but the *button* gate fired, not a residual-garment gate |
| C4 | Intelligent defect recovery | **PARTIAL** | Levels 1, 2, 5 implemented. Levels 3–4 (masked generative repair, controlled regeneration, retry budget) **not implemented** |
| C5 | Natural photographic appearance | **PARTIAL** | Protections preserved, restoration disabled, but **no quantitative plastic-skin/over-smoothing metric** exists. ArcFace alone is explicitly insufficient. |
| C6 | Studio pose compliance | **PARTIAL** | Identity-safe roll correction preserved. Gaze is a nose-offset proxy, **not** gaze; 3D frontalization deferred to V13.4 per spec |
| C7 | No background rectangles / visible torso boundaries | **PARTIAL** | Gated on injections and on executed cases; not validated across an unseen matrix |

## D. What would change the verdict

1. Obtain ≥50 authorized unseen portraits and ≥3 unseen uniform references;
   run `scripts/run_production_acceptance.py` to completion (≈6–8 min/case).
2. Perform human visual acceptance on every delivered output; record per-image
   accept/reject with reasons.
3. Add a gate for coloured (not merely white) edge halos — B4's defect.
4. Resolve or document C2's UNCERTAIN attributes.
5. Re-measure performance against the <60 s target, or revise the target.

Until 1–3 are done, the honest status is **NOT PRODUCTION READY**. The
pipeline is, on current evidence, suitable for supervised pilot use where every
delivery is human-reviewed before it reaches a customer.
