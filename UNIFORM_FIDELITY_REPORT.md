# Uniform Fidelity Report (V13.3.1)

Scope: how candidate garments are compared against an arbitrary target uniform
reference, what is genuinely measured, and what is **not** verified.

---

## 1. What V13.3 actually did

`UniformReferenceAnalyzer` advertises "15 dynamic attributes, zero hardcoded
templates". The analyzer does extract many attributes dynamically, but the
**validators that consume them** were hardcoded:

| Attribute | V13.3 behaviour | Consequence |
| :--- | :--- | :--- |
| Collar colour | `if collar_color == "red"` → fixed RGB window; `elif == "navy blue"` → fixed HSV window; `else` → `{"status": "PASS", "score": 1, "threshold": 1}` | Every colour except red and navy was **passed without measurement**. |
| Button colour | only `if expected_btn == "black"`, by counting pixels >195 in a 60 px strip | Counted fabric, not buttons; white/gold/matched buttons never checked. |
| Collar registration | red-pixel centroid, red collars only | Dragged off-axis by red sleeve trim; other colours unchecked. |
| Vest | `mean_torso > 175` | A light-coloured vest reads as "missing vest". |

So "universal compatibility" was asserted on the strength of two colours.

## 2. What V13.3.1 measures

**Colour comparison is perceptual and reference-driven.** The analyzer records
a measured CIELAB value for the collar (`collar_lab`), the body fabric
(`body_lab`) and the buttons (`button_lab`); validators compare the candidate
against those values by ΔE76.

- **Collar colour**: ΔE ≤ 28 inside the anatomical collar ROI. The tolerance
  absorbs fabric shading, studio white balance and JPEG drift while still
  separating one garment colour from another. Verified on arbitrary colours:
  green-vs-green PASS, green-vs-maroon FAIL, shaded green still PASS.
- **Collar colour extraction** uses `_distinctive_lab()`: the cluster of the
  collar band furthest from the body colour that still covers ≥10% of the band.
  A plain dominant-colour estimate returns the *shirt*, because the collar band
  of a garment photo contains more shoulder fabric than collar — measured on
  REF001, the dominant cluster was the light-blue gingham, not the red Peter
  Pan collar that defines the design.
- **Collar registration** is a left/right symmetry measure about the facial
  axis within the neck corridor, excluding lateral sleeve zones.
- **Buttons** are blobs, not bright pixels: CIELAB deviation from the placket's
  own median cloth colour, filtered by size, roundness, solidity and centreline
  proximity, sampled at the blob interior, compared by ΔE ≤ 40.
- **Fabric pattern** continues to use torso texture standard deviation
  (unchanged; see limits below).

## 3. Reference analysis confidence

Measured across the four available references:

| Reference | Collar colour (measured RGB) | ΔE collar↔body | Buttons (measured RGB) | Button blobs |
| :--- | :--- | ---: | :--- | ---: |
| REF001_red_collar | (208, 17, 50) red | 109.6 | (27, 29, 33) near-black | 5 |
| REF002_navy_collar | (26, 35, 66) navy | 171.2 | not measurable | 0 |
| REF003_tie | (204, 207, 232) white | 15.8 | (227, 229, 249) | 3 |
| REF004_vest | (191, 208, 237) light | 161.1 | not measurable | 1 |

Two observations that are reported rather than hidden:

1. **REF002 and REF004 button colour is not measurable** by the current blob
   criterion. Those cases now return **UNCERTAIN** and raise a warning, instead
   of the V13.3 behaviour of silently passing. Uncertain is the honest result;
   it is not a pass and must not be read as one.
2. **REF003's collar does not contrast with its body** (ΔE 15.8 — a white
   collar on a white shirt). Collar symmetry therefore returns UNCERTAIN for
   that reference rather than guessing.

## 4. Explicit UNCERTAIN semantics

Any attribute that cannot be reliably measured returns
`{"status": "UNCERTAIN"}` and adds a warning. UNCERTAIN is **not** counted as a
pass and **not** counted as a defect. It means the gate did not verify the
attribute, and a human reviewer must. This is the direct remedy for the
unmeasured-`PASS` pattern in §1.

## 5. Not verified — stated plainly

- **Genuinely unseen uniform designs have not been tested.** Only the four
  references in `test_dataset/uniform_references/` exist in this workspace.
  Arbitrary-colour support is demonstrated on synthetic fixtures
  (`tests/test_v13_3_1_fidelity_universal.py`), which proves the *code path* is
  colour-agnostic but does **not** prove performance on real unseen garments.
  **No claim of universal compatibility is made.**
- Striped and textured fabrics are not separately validated; only `checked` has
  a measured path, and pattern scale/frequency comparison between reference and
  candidate is still only a texture standard-deviation proxy.
- Pocket, badge, tie and vest structure comparison remains at V13.3 logic and
  retains hardcoded thresholds (e.g. `mean_torso > 175` for vests). These were
  out of scope for V13.3.1 and are recorded in REMAINING_ISSUES.md.
