# V13.2 Unseen Generalization Regression Test Report

## Overview
Comprehensive regression evaluation across unseen real-world portraits, challenging conditions, multiple uniform categories (red collar, navy collar, necktie, vest), and multiple background colors.

## Results Summary

| Test ID | Category | Scenario Description | Source Portrait | Uniform Reference | Background | Status | Biometric Sim | Geometry RMSE | Defects | Delivery Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T001** | Non-standard Hair | Boys with straight and curly hairstyles | `curly_hair.png` | `REF001_red_collar.png` | `#FFFFFF` | **completed** | **0.9980** | **0.0021** | None | **Delivered** |
| **T006** | Braided Silhouettes | Girls with different braid styles | `braids.png` | `REF001_red_collar.png` | `#FFFFFF` | **completed** | **0.9979** | **0.0010** | None | **Delivered** |
| **T011** | Hair Accessories | Girls with coloured bows and ribbons | `maroon_bows.png` | `REF001_red_collar.png` | `#FFFFFF` | **completed** | **0.9970** | **0.0010** | None | **Delivered** |
| **T016** | Complex Accessories | Girls with single braid and floral bow | `no_bows.png` | `REF001_red_collar.png` | `#FFFFFF` | **completed** | **0.9968** | **0.0013** | None | **Delivered** |
| **T021** | Facial Jewelry | Glasses, bindi and earrings | `glasses_bindi.png` | `REF002_navy_collar.png` | `#047EF6` | **completed** | **0.9921** | **0.0017** | None | **Delivered** |
| **T026** | Casual Source | Different original garments (plaid/casual) | `casual_shirt.png` | `REF003_tie.png` | `#77BF91` | **completed** | **0.9920** | **0.0013** | None | **Delivered** |
| **T031** | Harsh Lighting | Strong sunlight and deep facial shadows | `sunlight.png` | `REF001_red_collar.png` | `#ADD8E6` | **completed** | **0.9978** | **0.0025** | None | **Delivered** |
| **T036** | Low Resolution | Low-resolution and blurred photographs | `blurred.png` | `REF001_red_collar.png` | `#D3D3D3` | **completed** | **0.9986** | **0.0020** | None | **Delivered** |
| **T041** | Historical Print | Old and scanned photographs | `scanned.png` | `REF004_vest.png` | `#F5E6D3` | **completed** | **0.9900** | **0.0044** | None | **Delivered** |
| **T046** | Natural BG & Tilt | Complex backgrounds and tilted faces | `complex_background.png` | `REF002_navy_collar.png` | `#047EF6` | **completed** | **0.9951** | **0.0033** | None | **Delivered** |

## Production Quality Assurance Policy
- **Zero False Deliveries**: Any candidate with detected defects is strictly rejected and flagged for review.
- **Biometric Integrity**: Facial identity and geometry strictly preserved above threshold.
- **Artifact Protection**: Hair bows, ribbons, glasses, and bindi preserved without erasure.
