# CNN-LSTM RUL Predictor ¡ª C-MAPSS FD003

- **Task:** Remaining Useful Life (RUL) prediction for turbofan engines
- **Data:** NASA C-MAPSS FD003
- **Architecture:** Conv1d(32¡ú64) + BiLSTM(64¡Á2) + MLP(32¡ú1)
- **Input:** 50¡Á14 sensor sequence
- **RUL Cap:** 130 cycles (piecewise linear)
- **Test RMSE:** 22.0 cycles
- **Test NASA Score:** 22
- **Weight SHA256:** 00979451a180d31c1b29e7331f85d25ed4a35066268a99254e788b3fc3b81f06
- **Limitations:**
  - C-MAPSS is SIMULATED data ¡ª NOT real engine telemetry
  - Anonymous sensors (s1-s21) ¡ª do NOT map to physical quantities
  - Only FD001 has C=1 operating condition; others have C=6
  - Uncertainty: ensemble of 5 seeds recommended for production
- **Status:** experimental ¡ª validated_public_domain (C-MAPSS only)
