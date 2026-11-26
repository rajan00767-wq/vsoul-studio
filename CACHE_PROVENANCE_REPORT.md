# V13.2 Cache Policy & Provenance Enforcement Report

## Executive Summary
This report documents the remediation and independent verification of the `force_regenerate` cache policy and mandatory cryptographic provenance enforcement for external candidates in V13.2.

- **`force_regenerate` Policy Enforcement**: Verified that `raw_diffusion_cache_policy == "force_regenerate"` unconditionally bypasses explicit candidate inputs, authoritative disk cache lookups, forces fresh generative diffusion, and sets `cache_hit=False`, `fresh_inference=True`.
- **13-Field Provenance Manifest**: Enforced strictly in production mode. Unverifiable or missing manifests prevent delivery and force fresh inference.
- **Automated Regression Suite**: 5/5 unit tests passed in `tests/test_force_regenerate_policy.py`.

---

## 1. `force_regenerate` Policy Implementation

In `pipelines/uniform_swap/orchestrator.py` (lines 434–468):
```python
if config.raw_diffusion_cache_policy == "force_regenerate":
    cached_raw_candidate = None
    raw_vae_img = None
    cache_hit = False
    fresh_inference = True
    logger.info("[ORCHESTRATOR] Cache policy is 'force_regenerate': strictly ignoring explicit cached_raw_candidate and cache lookups. Forcing fresh generative inference.")
```
This guarantees that explicit parameters (`cached_raw_candidate`) cannot bypass the user's intent to regenerate.

---

## 2. Authoritative 13-Field Cryptographic Provenance Manifest

Every candidate stored in authoritative cache or accepted as an external candidate in production mode must include the following 13 cryptographically verified fields:

| Field # | Manifest Key | Description | Validation Rule |
|---|---|---|---|
| 1 | `original_source_sha256` | SHA256 of raw uploaded portrait | Mandatory SHA256 string |
| 2 | `normalized_source_sha256` | SHA256 of normalized portrait | Must match canonical normalized source |
| 3 | `uniform_ref_sha256` | SHA256 of uniform reference image | Strict equality assertion |
| 4 | `conditioning_image_sha256` | SHA256 of prepass conditioning | Mandatory SHA256 string |
| 5 | `effective_prompt_sha256` | SHA256 of full effective Qwen prompt | Prompt integrity assertion |
| 6 | `negative_prompt_sha256` | SHA256 of negative prompt | Negative prompt integrity assertion |
| 7 | `model_version` | Base model checkpoint identity | String match (`qwen-image-edit-2511-q4_k_m`) |
| 8 | `lora_version` | LoRA adapter identity & rank | String match (`lightning-4step-rank64`) |
| 9 | `inference_resolution` | Exact generation resolution `[w, h]` | Must equal `[800, 1024]` |
| 10 | `seed` | RNG generation seed | Integer |
| 11 | `steps` | Lightning diffusion steps | Integer (`4`) |
| 12 | `guidance_parameters` | CFG and identity lock settings | Dict (`cfg`, `latent_identity_lock`) |
| 13 | `candidate_image_sha256` | Cryptographic SHA256 of candidate PNG | Strict byte hash equality |

---

## 3. Production vs Diagnostic Validation Rules

1. **Production Mode (`diagnostic_mode=False`, default)**:
   - Manifest file (`.json`) must exist alongside candidate image.
   - All 13 fields must be present and non-empty.
   - `uniform_ref_sha256` must equal the expected reference SHA256.
   - `candidate_image_sha256` must equal the candidate's actual pixel hash.
   - Matching uniform color or type is **never** treated as proof of provenance.
   - Any violation strictly rejects candidate and triggers fresh diffusion.

2. **Diagnostic Mode (`diagnostic_mode=True`)**:
   - Only enabled for offline regression diagnostics.
   - Logs explicit warning that candidate lacks full provenance.

---

## 4. Test Verification Results

Automated regression suite `tests/test_force_regenerate_policy.py`:
- `test_force_regenerate_cannot_be_bypassed_by_explicit_candidate`: **PASS**
- `test_production_accepts_verified_provenance_manifest`: **PASS**
- `test_production_rejects_candidate_without_manifest`: **PASS**
- `test_production_rejects_mismatched_reference_provenance`: **PASS**
- `test_diagnostic_mode_permits_legacy_candidate`: **PASS**
