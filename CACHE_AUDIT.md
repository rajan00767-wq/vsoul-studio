# Authoritative Cache Audit & State Invalidation Report (Run 20261126_132126)

## 1. Root Cause of Previous Reference Pollution
In V12.8, loose filename globbing caused candidates from earlier runs to be reused for mismatched uniform templates.

## 2. V13.0 Cryptographic Cache Architecture (`UniformCacheManager`)
- **Cache Key Schema**:
  - `source_sha256`: SHA-256 of normalized input portrait
  - `uniform_ref_sha256`: SHA-256 of actual uniform reference template
  - `model_version`: Qwen-Image-Edit-2511-Q4_K_M
  - `lora_version`: Lightning-4steps-V1.0-bf16
  - `prompt_sha256`: SHA-256 of complete generation prompt
  - `pipeline_version`: `"v12.9"` / `"v13.0"`
  - `inference_params`: steps, cfg, seed
- **Candidate Validation**:
  Candidates are verified by `validate_external_candidate()` inspecting uniform pixel attributes against reference types before loading.
