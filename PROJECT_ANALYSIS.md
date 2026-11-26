# Vsoul Studio Project Analysis

## Static Prompt Fix Summary

**Problem:** Face and hair details missing every generation in static_test mode uniform swaps.

**Root Cause:** The `_build_universal_static_uniform_prompt` function creates excellent HEAD LOCK/HAIR LOCK prompt text (lines 1081-1089 of qwen_edit_pipeline.py), but this prompt is NEVER actually applied in the static_test code path. The VL analysis runs, but its results don't include the static prompt's identity/hair preservation elements.

**Fix:** Three code changes in `pipelines/qwen_edit_pipeline.py` that add ~0 seconds processing time:

### Change 1: After VL analysis in static_test mode
After the vl_plan is built (after line 8617), add static prompt integration that:
- Calls `_build_universal_static_uniform_prompt` to get the prompt text
- Flags vl_plan with `head_lock_enabled`, `hair_lock_enabled`, `no_beautification`, `accessories_preserve`
- Logs what's being enforced

### Change 2: After CFG scale (line ~8650)
Add static mode CFG scale guard and dimension fallback:
- Set `static_cfg_scale = 0.8` for more source-faithful generation
- If detail < 360, fall back to 1216px bucket even in static mode
- Add "static_mode_head_hair_locks_enforced" to step_reasons

### Change 3: After step reasons (line ~8684)
Inject the static prompt's key text into the generation conditioning:
- Construct full prompt with HEAD LOCK/HAIR LOCK elements
- Combine with VL-derived prompt
- Store in `vl_plan["full_generation_prompt"]`

**Result:** The HEAD LOCK/HAIR LOCK preservation text is now part of the Qwen conditioning, ensuring face and hair geometry are preserved without adding any prepass time or extra inference steps.

**Files to modify:** `pipelines/qwen_edit_pipeline.py`
**Lines to modify:** ~8617, ~8650, ~8684