# Full Test Failure Triage (V13.3.2)

Classification of all **71** pre-existing failures in
`python -m unittest discover -s tests` (513 tests, 57 failures, 14 errors,
1 skipped). None was introduced by V13.3.1 — the failing set is identical
before and after that work, verified by set comparison.

**No test was deleted, skipped, or weakened to improve the pass rate.**

---

## 1. Headline finding: none of the 71 is on the production path

The production entry point is `UniformSwapOrchestrator.run()`
(`pipelines/uniform_swap/orchestrator.py`). It touches the legacy pipeline at
exactly one place:

```
orchestrator.py:504   from pipelines.qwen_edit_pipeline import qwen_service
uniform_generator.py:341   qwen_pipeline_instance.qwen_edit_enhancer(...)
```

So production uses **`qwen_edit_enhancer()`** and nothing else from the legacy
module. Measured by AST analysis (`scratch/reach2.py`):

| Module | Imported where | On production path? |
| :--- | :--- | :--- |
| `pipelines/uniform_composite.py` | only inside `qwen_vl_image_edit_uniform_swap()` | **No** |
| `pipelines/uniform_review.py` | only inside `qwen_vl_image_edit_uniform_swap()` | **No** |
| `pipelines/uniform_matte` | not imported at all | **No** |
| module-level imports of either | none | — |
| `qwen_edit_enhancer()` calls to composite/review helpers | none | — |

`qwen_vl_image_edit_uniform_swap` is the route CLAUDE.md describes as the old
product, and `qwen_edit_uniform_swap` at the end of the same file is already
marked legacy and unreachable.

## 2. Classification

| Category | Count | Meaning |
| :--- | ---: | :--- |
| **A — Outdated test expectation (source-text assertion)** | **54** | Asserts that specific literal strings appear in `pipelines/qwen_edit_pipeline.py`. Tests code *text*, not behaviour. |
| **B — Legacy implementation defect (behavioural)** | **11** | Genuine behavioural assertions, all against legacy-only modules. |
| **C — Missing name / import error** | **6** | `NameError` / `ValueError` from tests that `exec()` a function out of the AST without its dependencies. |
| **D — Genuine production-path defect** | **0** | — |

### Category A — 54 outdated test expectations

These parse `pipelines/qwen_edit_pipeline.py` with `ast` and assert on literal
substrings. Example (`test_repeated_inputs_use_a_content_stable_seed`):

```python
source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
self.assertNotIn('zlib.crc32(job_id.encode("utf-8"), uniform_seed)', source)
self.assertIn('"seed_policy": "content_and_generation_contract_stable_v22_compact_uniform_refine"', source)
```

They fail because the strings were renamed or the code was restructured, not
because behaviour regressed. A test of this shape cannot distinguish a real
regression from a rename, which is why 54 of them drifted silently.

Modules: `test_uniform_prompt` (42), `test_uniform_dynamic_only` (10),
`test_uniform_arm_guide` (1), `test_upload_alpha` (1).

### Category B — 11 legacy behavioural failures

Real assertions with real measurements, but every target module is
legacy-only:

| Test | Target module | Symptom |
| :--- | :--- | :--- |
| `CompositeTests.test_close_up_face_preserves_uploaded_body_width` | `uniform_composite` | `[5, 65, 95, 140] != [5, 80, 95, 140]` |
| `CompositeTests.test_rough_fit_does_not_extend_skin_scan_through_collar_opening` | `uniform_composite` | colour equality assertion |
| `CompositeTests.test_rough_fit_does_not_paint_a_connector_to_disconnected_dark_patch` | `uniform_composite` | `(12,28,50) != (4,126,246)` |
| `CompositeTests.test_shoulders_cropped_on_one_side_keep_their_real_width` | `uniform_composite` | `86.0 < 90.0` — clipped shoulder rows discarded |
| `UniformMatteTests.test_strict_flat_background_keeps_blue_clothing_and_pale_accessory` | `uniform_matte` | assertion False |
| `GateHonestyTests.test_bypass_log_no_longer_claims_a_bypass` | `uniform_review` | log wording |
| `LivePromptWiringTests.test_neck_lock_is_built_and_used` | legacy prompt | `1 < 2` |
| `SourceGarmentRemovalTests.*` (2) | legacy prompt | clause text |
| `WiringTests.test_edge_instruction_survives_prompt_budget_trimming` | legacy prompt | `EDGE LOCK` absent |
| `PlaitStructureTests.test_tied_hair_recovery_reaches_the_measured_endpoint` | legacy prompt | marker absent |

Note `test_shoulders_cropped_on_one_side_keep_their_real_width` is *conceptually*
the same failure mode V13.3.1 fixed in the production shoulder gate — clipped
rows being discarded. The legacy `uniform_composite` implementation still has
it. Since that module is unreachable from production, fixing it is cleanup,
not a defect fix, and it is listed in REMAINING_ISSUES.md.

### Category C — 6 import/name errors

Tests that `exec()` a function extracted from the AST into a namespace missing
its dependencies:

- `NameError: name 'QwenEditPipeline' is not defined` (5 ×
  `MeasureExposedNeckTests`)
- `NameError: name 'fit_prompt_to_token_budget' is not defined`
- `NameError: name 're' is not defined`
- `ValueError: substring not found` (the test slices source by a marker string
  that no longer exists)

These are harness defects in the tests themselves.

## 3. Why these were not fixed in V13.3.2

Deliberate, and stated rather than quietly skipped:

1. **Zero production-path impact.** All 71 exercise code the delivered
   pipeline never executes.
2. **Category A has negative value as written.** Rewriting 54 source-text
   assertions into behavioural tests of a legacy route would be substantial
   work on code scheduled for removal, and re-pinning them to the *current*
   strings would re-create the same brittleness.
3. **The V13.3.2 scope is delivered-image quality.** Effort went to the three
   confirmed delivery defects (PA_11 halo, PA_02 seam, PA_03 bow), which
   affect real customer output.

## 4. Recommendation

- **Category A**: delete or convert. A test asserting on source text should be
  replaced by one asserting on the function's output, or removed with the
  legacy route.
- **Category B**: fix with the legacy route if it is kept; otherwise retire
  alongside it.
- **Category C**: repair the `exec()` namespaces, cheap and self-contained.
- **Until then**: treat 71 as the known baseline. Any change to that number,
  up or down, is a signal worth investigating. V13.3.2 re-verifies it by set
  comparison on every full-suite run.

## 5. Reproduction

```
.venv/Scripts/python.exe -m unittest discover -s tests
.venv/Scripts/python.exe scratch/triage_classify3.py   # category breakdown
.venv/Scripts/python.exe scratch/reach2.py             # production reachability
```
