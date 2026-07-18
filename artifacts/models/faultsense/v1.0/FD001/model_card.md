# FaultSense LSTM-Autoencoder ¡ª C-MAPSS FD001

- **Task:** Anomaly detection + RUL prediction (LSTM Autoencoder)
- **Data:** NASA C-MAPSS FD001
- **Architecture:** 2-layer LSTM Encoder (hidden=32) + 1-layer Decoder + RUL MLP
- **Input:** 30¡Á14 sensor sequence
- **Anomaly:** Reconstruction error threshold (k-sigma, k=2.5, calibrated on early normal cycles)
- **RUL Cap:** 130
- **Test RMSE:** 16.1, **NASA:** 6
- **Status:** experimental ¡ª validated_public_domain (C-MAPSS only)
