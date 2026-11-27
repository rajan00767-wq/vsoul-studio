# V13.3.2 Fresh Inference & Production Acceptance Report

**Date**: 2026-11-27  
**Branch**: `v13.3-production-stabilization`  
**Execution Environment**: Windows 11, Python 3.10, PyTorch 2.5.1+cu124, NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM)  
**Verification Standard**: Verified Fresh Inference (`provenance=fresh_inference_verified`, `raw_diffusion_cache_policy="force_regenerate"`, `diagnostic_mode=False`, `invocations=1`)  

---

## 1. Executive Summary

In accordance with V13.3.2 Release Closure requirements, all seven mandatory production acceptance cases have been executed under genuine, forced GPU regeneration. No cached diffusion outputs, cache replays, or repaired historical artifacts were substituted.

Every generated output was independently validated by the authoritative 22-gate production quality validator (`pipelines/uniform_swap/quality_validator.py`).

**Key Results**:
- **Execution Pass Rate**: **7 / 7 (100%)** successfully completed and delivered with **0 defects**.
- **Average Latency**: **386.2 seconds** (~6.4 minutes per portrait).
- **Peak VRAM**: **7,014.7 MB – 7,034.6 MB** (well within the 8,192 MB hardware budget, 0 CUDA OOM).
- **Zero Hallucination / Identity Preservation**: Protected facial corridor preserved with straight-RGBA compositing; identity similarity > 0.985 across all cases.
- **Visual Comparison Sheets**: 4-panel visual comparison sheets generated for all 7 cases and archived in `outputs/production_acceptance/comparisons/`.
- **Production Status**: Automated quality gates **PASS (100%)**; formal release remains **strictly pending human visual acceptance** by the client.

---

## 2. Fresh Inference Benchmark Results

| Case ID | Execution ID | Source Portrait | Uniform Reference | Provenance | Invocations | Total Latency (s) | Diffusion (s) | Peak VRAM (MB) | Defects | Delivery Status |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PA_01_Curly_RedCollar** | `PA_01_Curly_RedCollar-ab7af30926fa` | `curly_hair.png` | `REF001_red_collar.png` | `fresh_inference_verified` | 1 | 325.51 | 284.02 | 7015.1 | None | **COMPLETED** |
| **PA_02_Braids_NavyCollar** | `PA_02_Braids_NavyCollar-9018a6902600` | `braids.png` | `REF002_navy_collar.png` | `fresh_inference_verified` | 1 | 419.52 | 395.92 | 7024.5 | None | **COMPLETED** |
| **PA_03_MaroonBows_Tie** | `PA_03_MaroonBows_Tie-317c96e51d33` | `maroon_bows.png` | `REF003_tie.png` | `fresh_inference_verified` | 1 | 572.64 | 550.62 | 7034.6 | None | **COMPLETED** |
| **PA_04_NoBows_Vest** | `PA_04_NoBows_Vest-1c438a3d485f` | `no_bows.png` | `REF004_vest.png` | `fresh_inference_verified` | 1 | 330.40 | 289.49 | 7015.9 | None | **COMPLETED** |
| **PA_05_GlassesBindi_RedCollar** | `PA_05_GlassesBindi_RedCollar-e21b7f755cd2` | `glasses_bindi.png` | `REF001_red_collar.png` | `fresh_inference_verified` | 1 | 378.84 | 336.73 | 7014.9 | None | **COMPLETED** |
| **PA_11_ShortBoy_Tie** | `PA_11_ShortBoy_Tie-4abe22a0031f` | `blue_bg_boy.png` | `REF003_tie.png` | `fresh_inference_verified` | 1 | 330.52 | 290.10 | 7014.7 | None | **COMPLETED** |
| **PA_12_StraightHair_RedCollar** | `PA_12_StraightHair_RedCollar-730187d05d84` | `straight_hair.png` | `REF001_red_collar.png` | `fresh_inference_verified` | 1 | 346.05 | 302.58 | 7015.9 | None | **COMPLETED** |

*Raw benchmark log stored in [FRESH_INFERENCE_BENCHMARK.csv](file:///C:/AI/vsoul-studio/FRESH_INFERENCE_BENCHMARK.csv) and [provenance_manifest.jsonl](file:///C:/AI/vsoul-studio/outputs/production_acceptance/provenance_manifest.jsonl).*

---

## 3. Targeted Defect Verification & Root-Cause Resolution

### Case 1: PA_11 — Universal Coloured Hair-Crown Halo
- **Previous Defect**: V13.3.1 delivered with 841 px of saturated cyan/blue background spill contaminating hair boundary pixels along the scalp crown (`RGB [0, 129, 253]`).
- **V13.3.2 Resolution**: Introduced `pipelines/uniform_swap/edge_decontamination.py`. Utilizes CIELAB chroma projection to distinguish between genuine hair pixels and ambient boundary spill, decontaminating fringing pixels via local foreground unmixing.
- **Verification Result**: Halo area dropped to **0 px**. Gate 21 (`detect_colored_edge_halo`) confirmed clean boundary without hair detail erosion.

### Case 2: PA_02 — Internal Rectangular Chest Seam
- **Previous Defect**: Torso inpainting in `compositor.py` produced a noticeable horizontal line at $y \approx 840\text{--}886$ due to flat rectangular body gap patching.
- **V13.3.2 Resolution**: Replaced the flat box cutoff in `identity_extractor.py` with an anatomically curved parabolic neckline reaching $0.26 \cdot f_h$ below chin, coupled with gradient-feathered transition blurring in `compositor.py`.
- **Verification Result**: Gate 20 (`rectangular_garment_boundary`) confirmed step gradient variance below threshold (0 horizontal step artifacts detected).

### Case 3: PA_03 — Missing Maroon Bow (Accessory Erasure)
- **Previous Defect**: Left bow erased during diffusion compositing. Accessory detector dropped lower chest-level ribbons ($y > \text{chin} + 0.28\cdot f_h$), and validator aggregated pixels globally rather than per-instance.
- **V13.3.2 Resolution**: 
  - Expanded lower torso search zone for braid ribbons and accessories.
  - Added individual instance tracking (`individual_bows`) in `identity_extractor.py`.
  - Enforced per-instance bilateral survival ($\ge 70\%$ retention per bow) in `quality_validator.py`.
- **Verification Result**: Both left and right bows retained post-compositing (left: 3,662 px, right: 5,247 px). Gate 7 passed.

### Case 4: PA_01, PA_05, PA_12 — Stale Button-Detector False Rejections
- **Previous Defect**: In `reference_analyzer.py`, dense check/gingham fabric patterns triggered false button detections, leading to false `incorrect_button_color` rejections.
- **V13.3.2 Resolution**: Implemented component density filtering (`effective_min_area = max(min_area, 60.0)` when $n > 12$) and vertical spacing checks. Returns `UNCERTAIN` for concealed or subtle buttons on textured fabrics.
- **Verification Result**: Verified clean delivery across PA_01, PA_05, and PA_12 with 0 false button rejections.

### Case 5: PA_04 — Neutral/Achromatic Edge Contamination False Positive
- **Previous Defect**: White/grey backdrops with near-zero chroma triggered false positive colored halo alarms due to noisy unit chroma vectors.
- **V13.3.2 Resolution**: Added `src_chroma < 10.0` check in `edge_decontamination.py`, correctly recognizing neutral source backdrops and bypassing chroma projection.
- **Verification Result**: Clean delivery with 0 defects.

---

## 4. Visual Comparison Panels (4-Panel Inspection)

Visual inspection sheets have been built showing:
1. Panel 1: Original Source Portrait
2. Panel 2: Target Uniform Reference
3. Panel 3: Previous V13.3.1 Output
4. Panel 4: New V13.3.2 Fresh-Inference Output

- [PA_01_Curly_RedCollar_v13_3_2_comparison.png](file:///C:/AI/vsoul-studio/outputs/production_acceptance/comparisons/PA_01_Curly_RedCollar_v13_3_2_comparison.png)
- [PA_02_Braids_NavyCollar_v13_3_2_comparison.png](file:///C:/AI/vsoul-studio/outputs/production_acceptance/comparisons/PA_02_Braids_NavyCollar_v13_3_2_comparison.png)
- [PA_03_MaroonBows_Tie_v13_3_2_comparison.png](file:///C:/AI/vsoul-studio/outputs/production_acceptance/comparisons/PA_03_MaroonBows_Tie_v13_3_2_comparison.png)
- [PA_04_NoBows_Vest_v13_3_2_comparison.png](file:///C:/AI/vsoul-studio/outputs/production_acceptance/comparisons/PA_04_NoBows_Vest_v13_3_2_comparison.png)
- [PA_05_GlassesBindi_RedCollar_v13_3_2_comparison.png](file:///C:/AI/vsoul-studio/outputs/production_acceptance/comparisons/PA_05_GlassesBindi_RedCollar_v13_3_2_comparison.png)
- [PA_11_ShortBoy_Tie_v13_3_2_comparison.png](file:///C:/AI/vsoul-studio/outputs/production_acceptance/comparisons/PA_11_ShortBoy_Tie_v13_3_2_comparison.png)
- [PA_12_StraightHair_RedCollar_v13_3_2_comparison.png](file:///C:/AI/vsoul-studio/outputs/production_acceptance/comparisons/PA_12_StraightHair_RedCollar_v13_3_2_comparison.png)

---

## 5. Non-Destructive Quality Gate Verification

The verification script `scripts/verify_quality_gates.py` was updated to separate human-authored documentation from machine-generated test runs:
- Preserved Engineering Audit: [QUALITY_GATE_REGRESSION_REPORT.md](file:///C:/AI/vsoul-studio/QUALITY_GATE_REGRESSION_REPORT.md) (remains unmodified).
- Generated Markdown Summary: [QUALITY_GATE_TEST_SUMMARY.md](file:///C:/AI/vsoul-studio/QUALITY_GATE_TEST_SUMMARY.md).
- Generated Machine-Readable JSON: [QUALITY_GATE_TEST_SUMMARY.json](file:///C:/AI/vsoul-studio/QUALITY_GATE_TEST_SUMMARY.json).
- Timestamped Execution Run: `reports/quality_gate_runs/`

**Test Results Across All 22 Mandatory Quality Gates**:
- Total Conditions: 44 (22 True Positive Injections, 22 True Negative Compliant Outputs)
- Accuracy: **100.0%** (44/44)
- Precision: **100.0%**
- Recall: **100.0%**
- Specificity: **100.0%**
- Delivery Rejection on Injection: **100.0%** (22/22)

---

## 6. Release Verdict

| Milestone | Status | Notes |
| :--- | :---: | :--- |
| **All 22 Quality Gates Verified** | **PASS** | 44/44 tests passed with zero false alarms. |
| **Defect Recoveries Verified** | **PASS** | Halo, seam, bow, and button fixes proven on real fresh outputs. |
| **Fresh Inference Provenance** | **PASS** | 7/7 executed with forced regeneration (`invocations=1`). |
| **Memory / Stability Profile** | **PASS** | Peak VRAM ≤ 7034.6 MB (8 GB hardware), zero crashes/OOM. |
| **Non-Destructive Reporting** | **PASS** | Verified by `tests/test_non_destructive_reporting.py`. |
| **Human Visual Review** | **PENDING** | **Strictly pending customer visual review of comparison sheets.** |
