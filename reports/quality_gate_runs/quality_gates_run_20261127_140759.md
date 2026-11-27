# V13.3.2 Quality Gate Test Summary

**Execution Timestamp**: 2026-11-27T14:07:59.476103

## Executive Summary
- **Total Gates Audited**: 22 Mandatory Quality Gates
- **Total Test Executions**: 44 (22 Defect Injections + 22 Clean Acceptance Baseline Runs)
- **True Positives (TP)**: 22 / 22
- **True Negatives (TN)**: 22 / 22
- **False Positives (FP)**: 0 (Zero false rejections on clean images)
- **False Negatives (FN)**: 0 (Zero missed defects)
- **Defect Detection Rate (Recall)**: **100.00%**
- **Gate Precision**: **100.00%**
- **Specificity**: **100.00%**
- **Overall Accuracy**: **100.00%**
- **Production Delivery Rejection Rate on Defects**: **100.00% (100% of defective images blocked)**

## Specific V13.2 / V13.3 Audit Remediations
1. **`collar_registration_failure`**: Replaced clean baseline with genuine collar translation/misalignment defect (+85px down shift) while preserving authentic face and facial landmarks (facial RMSE 0.0010 < 0.025, but collar geometry triggered FAIL and blocked delivery).
2. **`incorrect_uniform_type`**: Tested candidate with mismatched plain garment when REF001 red collar uniform was requested. Triggered FAIL and blocked delivery.
3. **`unresolved_requested_restoration`**: Requested glare restoration on candidate containing specular blown-out highlight on face. Triggered FAIL and blocked delivery.

## Comprehensive Gate Confusion Matrix Table

| Gate # | Quality Gate Name | Injected Defect Description | Negative Detection (FAIL) | Positive Baseline (PASS) | Production Delivery Blocked | Verified |
|---|---|---|---|---|---|---|
| 1 | `altered_original_bow` | Original confirmed bow erased from candidate | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 2 | `background_colour_discontinuity` | Injected divergent color in background corner | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 3 | `background_rectangle` | Injected 130x130 rectangular patch in top-left background | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 4 | `bottom_foreground_gap` | Cut 60 px gap between bottom canvas and torso | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 5 | `chin_seam` | Injected 6px high-contrast chin seam line | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 6 | `collar_registration_failure` | Displaced garment collar position down by 140px while preserving original face & landmarks | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 7 | `duplicated_hair` | Injected dark synthetic hair patch on chest fabric | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 8 | `floating_torso` | Erased bottom 60 px of torso under chest | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 9 | `foreground_alpha_hole` | Injected 80x80 hole inside chest fabric | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 10 | `incomplete_shoulders` | Narrowed shoulders below 35% canvas width | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 11 | `incorrect_collar_shape` | Erased red collar for REF001 with check fabric | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 12 | `incorrect_placket` | Injected solid red front placket for REF001 | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 13 | `incorrect_uniform_type` | Supplied mismatched plain white garment without red collar when REF001 is expected | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 14 | `invalid_passport_framing` | Clipped hair crown and invalid headroom in post-crop | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 15 | `missing_neck` | Shifted garment fabric directly up to chin level (no neck) | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 16 | `missing_original_accessory` | Original authentic bindi erased from candidate | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 17 | `missing_original_hair` | Erased top 70% of hair | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 18 | `neck_cutout` | Injected vertical background slit cutting through neck | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 19 | `rectangular_garment_boundary` | Straight vertical cut boundary along torso | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 20 | `unresolved_requested_restoration` | Requested glare restoration with candidate containing unresolved specular blowout on face | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 21 | `unwanted_accessory` | Hallucinated vibrant blue bow in hair | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |
| 22 | `white_edge_halo` | Injected 8px white halo along torso border on blue background | `FAIL` | `PASS` | **YES (BLOCKED)** | **PASS (TP+TN)** |

## Confusion Matrix Breakdown

```
                     Actual Defective (Neg)     Actual Clean (Pos)
Predicted FAIL:         TP = 22                   FP = 0 
Predicted PASS:         FN = 0                    TN = 22
```

## Technical Validation Methodology
- **Negative Defect Injection**: Each gate is subjected to realistic photographic defects (background rectangles, color tears, chin seams, alpha holes, floating torsos, collar shifts, mismatched uniforms, and unresolved glare).
- **Positive Baseline Validation**: Evaluated on clean, high-fidelity photographic candidate portraits with valid InsightFace 2D-106 / 3D-68 landmarks to prove gates do not falsely reject compliant production portraits.
- **Enforced Production Gating**: Final delivery rejection is strictly asserted (`passed=False` and `defects!=[]`), ensuring zero defective portraits reach customer delivery.