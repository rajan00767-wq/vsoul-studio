# Recovery Effectiveness Report (V13.3.1)

Scope: `pipelines/uniform_swap/recovery_controller.py` — what the controller
repairs, what it declines, and what is not yet implemented.

---

## 1. Safety hierarchy

| Level | Mechanism | Implemented |
| :-- | :--- | :--- |
| 1 | Deterministic localized repair (halo de-fringing, neck shadow lift, protected-face restore) | **Yes** |
| 2 | Reference-guided localized reconstruction (contour-driven shoulder rebuild) | **Yes** (V13.3.1) |
| 3 | Masked generative repair | **No** — not implemented; see §5 |
| 4 | Controlled full garment regeneration | **No** — no retry budget wired; see §5 |
| 5 | Manual review (`needs_review`, delivery blocked) | **Yes** |

Localized recovery is bounded to **2 cycles** by default
(`RecoveryController(max_attempts=2)`).

## 2. Rollback — the substantive change

V13.3 accepted any attempt whose defect **count** fell. An attempt that cleared
two defects and introduced a third was therefore accepted, and shipped a defect
the candidate never had.

V13.3.1 rejects any attempt introducing a defect **type** absent from the
pre-repair set, and rolls back to the previous best candidate. Comparison uses
`_defect_key()`, which strips the measured numbers from a defect string
(`"missing_crown_bow_1 (559 px < 1524)"` → `"missing_crown_bow_1"`) so the same
defect at a slightly different measurement is not mistaken for a new one.

Verified by `tests/test_v13_3_1_recovery.py`:

- 2 defects → 1 *different* defect ⇒ **rolled back**, `defects_after == 2`,
  action log records the rollback, `success == False`.
- 1 defect → 0 defects ⇒ accepted, `success == True`.

## 3. Shoulder reconstruction

V13.3 pushed every torso row out to fixed canvas fractions (0.12 w / 0.88 w).
That is the rectangular expansion the specification prohibits: it squares the
silhouette, imposes one body shape on every subject, and on a rotated portrait
widens each side by a different wrong amount.

V13.3.1 exploits the fact that an outer shoulder contour is **convex** — it
widens as it descends, then flattens, and never narrows and widens again. The
repair takes the **lower convex hull** of each side's edge profile; rows
bulging inside that hull are the clipped stretch, and the hull supplies the
slope to restore them to. Fill is inpainted from surrounding garment so fabric
scale and sleeve geometry follow existing cloth rather than being stretched.

Two alternatives were implemented and rejected **by measurement**:

| Attempt | Why it failed |
| :--- | :--- |
| Running-extreme envelope | For a monotonically widening edge the running extreme always equals the current value, so it detects nothing (measured: 0 clipped rows on a 40-row cut). |
| Straight-line fit | A line through a contour that widens then plateaus predicts an ever-widening body, and "repairs" a completely intact silhouette (measured: 2364 px of spurious fill). |

Hull output versus ground truth on the test fixture: 75.0 vs 75, 58.8 vs 59,
94.8 vs 95 — within 1 px.

**Declines.** Reconstruction is refused, and the case routed to review, when:

- reach asymmetry about the face axis > 0.45 (a whole missing side has no
  intact neighbours to extrapolate from; inventing one is fabrication); or
- the deficit exceeds 18% of torso area (beyond a localized repair; inpainting
  that much produces duplicated fabric and visible seams).

**Detectable class, stated honestly**: this repairs contour *violations* —
flat cuts, canvas-edge clipping, matting failures. A cut running exactly
parallel to the shoulder slope leaves a silhouette that is still
self-consistently convex, and nothing in the alpha distinguishes it from a
genuinely slighter child. That case is not detectable from silhouette geometry
alone and is documented in the test file rather than papered over.

## 4. Post-recovery re-validation

Every attempt is re-validated through the **same** `validate_pre_crop` used for
the original gate, so identity, protected regions, uniform fidelity, foreground
alpha and framing are all re-checked. Protected facial pixels are restored from
the untouched source during garment recovery and are never modified by it.

## 5. Not implemented — stated plainly

- **Masked generative repair** (level 3) and **controlled full regeneration**
  (level 4) are not implemented. There is no configurable regeneration retry
  budget. A defect that deterministic and reference-guided repair cannot fix
  goes to `needs_review`.
- Defect-specific routes exist for: incomplete shoulders, neck seam / neck
  lighting mismatch, white edge halos, background fringe, identity drift.
  Routes **do not** exist for: incorrect button colour, incorrect button
  placement, collar misregistration, incorrect collar colour, minor collar
  shape defects, incorrect placket, residual original garment, hair-edge
  contamination, small foreground alpha holes. Those defects are detected and
  gated, but not repaired.
- **Effectiveness rates are not quoted.** Recovery did not trigger on the
  executed fresh-inference cases, so there is no measured
  repair-success-rate to report. Quoting one from synthetic injections alone
  would misrepresent it.
