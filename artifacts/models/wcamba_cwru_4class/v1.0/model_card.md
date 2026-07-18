# WCamba CWRU 4-Class Bearing Fault Classifier

- **Task:** Bearing fault classification (normal, inner_race, outer_race, ball)
- **Data:** CWRU Bearing Data Center, 12kHz drive-end
- **Architecture:** WideKernel 1D-CNN (64/32/64 conv + FC4)
- **Input:** 1024-point vibration windows, per-window z-score normalized
- **Output:** 4-class softmax probabilities
- **Test Accuracy:** 1.0000
- **Weight SHA256:** 4c5a7d01bf371eb6957ba35ecdbbad00e69cfa29f6f3698f9de7b79008779ca4
- **Limitations:** CWRU is lab bench data ¡ª NOT real aero-engine bearings.
  Cross-domain performance on PU/HIT datasets not evaluated.
  Does NOT include cage fault class.
- **Status:** experimental ¡ª validated_public_domain (CWRU only)
