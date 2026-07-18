# PatchCore ¡ª AeBAD-S Anomaly Detection

- **Task:** Unsupervised anomaly detection on aero-engine blades
- **Data:** AeBAD-S (single blade images)
- **Architecture:** WideResNet-50 (layer2+layer3) + Coreset Memory Bank
- **Image AUROC:** 0.6050
- **Best F1:** 0.6334
- **Limitations:** AeBAD has domain shift (illumination/view). Coreset sampling loses minority patterns.
- **Status:** experimental ¡ª validated_public_domain (AeBAD only)
