# V13.2 Unseen Generalization Regression Test Report

## Overview
Comprehensive regression evaluation across unseen real-world portraits, challenging conditions, multiple uniform categories (red collar, navy collar, necktie, vest), and multiple background colors.

## Results Summary

| Test ID | Category | Source | Uniform | Status | Biometric Sim | Geometry RMSE | Defects | Delivered |
|---------|----------|--------|---------|--------|---------------|---------------|---------|-----------|
| REG_01_Scanned_RedCollar | Scanned Vintage Photo | scanned.png | REF001_red_collar.png | **needs_review** | 0.9898 | 0.0032 | 3 | needs_review |
| REG_02_ComplexBG_NavyCollar | Complex Natural Background | complex_background.png | REF002_navy_collar.png | **needs_review** | 0.9950 | 0.0036 | 2 | needs_review |
| REG_03_GlassesBindi_Vest | Eyeglasses & Bindi Retention | glasses_bindi.png | REF004_vest.png | **needs_review** | 0.9945 | 0.0009 | 3 | needs_review |
| REG_04_LongBraids_Tie | Long Braids Over Garment | braids.png | REF003_tie.png | **needs_review** | 0.9961 | 0.0004 | 1 | needs_review |
| REG_05_ShortBoy_Tie | Short Hair Boy Portrait | blue_bg_boy.png | REF003_tie.png | **needs_review** | 0.9916 | 0.0017 | 2 | needs_review |
| REG_06_StraightHair_RedCollar | Straight Hair with Custom Backdrop | straight_hair.png | REF001_red_collar.png | **needs_review** | 0.9879 | 0.0013 | 4 | needs_review |
| REG_07_BowsRibbons_NavyCollar | Hair Bows & Ribbons | blue_bg_bows.png | REF002_navy_collar.png | **completed** | 0.9959 | 0.0015 | 0 | REG_07_BowsRibbons_NavyCollar_uniform.png |
| REG_08_Sunlight_Vest | Harsh Sunlight Balancing | sunlight.png | REF004_vest.png | **needs_review** | 0.9934 | 0.0013 | 4 | needs_review |

## Production Quality Assurance Policy
- **Zero False Deliveries**: Any candidate with detected defects is strictly rejected and flagged for review.
- **Biometric Integrity**: Facial identity and geometry strictly preserved above threshold.
- **Artifact Protection**: Hair bows, ribbons, glasses, and bindi preserved without erasure.
