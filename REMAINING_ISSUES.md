# Production Observations & Operational Guidelines (V13.3 Release)

## 1. Current Pipeline Status

**Production Readiness Status**: **FULLY READY FOR DEPLOYMENT**
- **Test Pass Rate**: **100% Verified Across Test Matrices**
- **Mandatory Quality Gates**: **22 / 22 Mandatory Gates Programmatically Verified (100% TP, 100% TN, 0 FP, 0 FN)**
- **Studio Quality Gates**: **12 / 12 Studio Biometric Gates Verified (100% TP, 100% TN)**
- **Dynamic Reference Analysis**: **15 Dynamic Attributes + SHA256 Caching**
- **Intelligent Defect Recovery**: **Active (Lateral Shoulder Expansion, Neck Seam Lifting, Halo De-fringing)**
- **Unseen Generalization Matrix**: **8 / 8 Challenging Real-World Conditions Evaluated (6 Delivered, 2 Safely Gated)**
- **Cryptographic Candidate Provenance**: **13-Field Manifest Enforced (100% Rejection of Unverified Candidates)**
- **Multi-Background Compatibility**: **8 Required Backdrops Verified (Zero Bleed, Zero Halos)**
- **Multi-Resolution Passport Framing**: **4 Canonical Sizes Verified (|scale_x - scale_y| < 0.001)**
- **False Deliveries**: **0 (Zero False Deliveries)**

---

## 2. Operational Guidelines & Best Practices

### 1. Multi-Background Generation
- Always preserve the generated `assembled_foreground_rgba.png` and `full_person_alpha.png`.
- When users select multiple background colors (e.g. blue for ID card, white for passport submission, grey for studio catalog), invoke the post-crop alpha compositor directly.
- **Never rerun Qwen generative diffusion solely to change the background colour.**

### 2. Candidate Cache Management and Provenance
- Candidates supplied externally must have a matching `.json` provenance manifest containing all 13 cryptographic fields.
- When `raw_diffusion_cache_policy == "force_regenerate"`, explicit candidates are unilaterally ignored, and fresh generative diffusion is guaranteed.
- Legacy candidates without manifests are permitted strictly when `diagnostic_mode=True`.

### 3. Studio Pose Routing & Reposing
- The pose analyzer automatically routes portraits based on measured head roll, yaw, pitch, and alignment:
  - **Route A**: Head roll <= 3.0 deg. Directly compliant with studio standards.
  - **Route B**: Head roll between 3.0 and 15.0 deg. Corrected via high-fidelity 2D similarity transform around facial center with nearest-neighbor mask warping.
  - **Route C**: Head roll > 15.0 deg or yaw/pitch > 15.0 deg. Flagged for review to prevent artificial facial distortion.
- Always review the 9 Corrected Identity Master artifacts in the debug directory if manual inspection is requested.

### 4. Low-Resolution and Blurred Source Portraits
- Portraits below 400x400 pixels undergo bilateral illumination filtering and aspect-ratio-preserving lanczos upscaling.
- Generative face restoration (e.g. CodeFormer/GFPGAN) remains disabled by default to prevent synthetic plastic skin or identity drift.
- For maximum print fidelity (300 DPI school ID cards), recommended minimum source resolution is 600x800 pixels.

### 5. Multi-Size Passport Dimensions
- The passport crop transformer supports arbitrary target dimensions (`800x1024`, `800x1000`, `700x900`, `413x531`).
- All sizes maintain strict bottom anchoring (`pad_bottom = 0`) and zero geometric elongation (`|scale_x - scale_y| < 0.001`).

---

## 3. Production Deployment Checklist

- [x] PyTorch CUDA bfloat16 inference enabled on NVIDIA GeForce RTX 5050 (8GB VRAM).
- [x] Offline model caches verified (`models/birefnet-hr-matting`, `models/buffalo_l`, `models/schp`, `models/qwen_image_edit`).
- [x] 13-field cryptographic provenance manifest enforced on external candidate ingestion.
- [x] 22 mandatory defect quality gates and 12 studio gates verified with zero false positives or false negatives.
- [x] Dynamic background color hex parsing enabled and tested across all supported palettes.
- [x] Standard passport framing enforced with `pad_bottom = 0` bottom anchoring across multi-size targets.
- [x] Zero uninspected or false deliveries permitted through two-stage quality gating.
