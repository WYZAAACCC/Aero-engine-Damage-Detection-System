"""CNN-LSTM RUL 预测器 + 简单基线 — P1_HIGH"""

import numpy as np

from ._base import AssetRunResult, ImplementationBase


class CNNLSTMRULPredictor(ImplementationBase):
    """RUL 预测器——当前为线性趋势基线。

    审计修复 (AER-001): 资产 ID 暗示 CNN-LSTM，实际实现为线性趋势。
    必须诚实标注。真正的 CNN-LSTM 需要 TensorFlow + C-MAPSS 训练权重。
    """

    asset_id = "reliability_model.rul.cnn_lstm_cmapss"

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        if not inputs:
            return {"ok": False, "issues": ["No sensor time series provided"]}
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        params = {**self.default_params(), **parameters}
        data = inputs[0] if inputs else {}
        import time
        t0 = time.time()

        # 获取传感器数据
        signals = {}
        for key in data:
            val = data[key]
            if isinstance(val, (list, np.ndarray)):
                signals[key] = np.asarray(val, dtype=np.float64)

        # ── 线性趋势基线（不是 CNN-LSTM）──
        best_sensor = None
        best_trend = 0
        for name, arr in signals.items():
            if len(arr) > 10:
                trend = np.polyfit(np.arange(len(arr)), arr, 1)[0]
                if abs(trend) > abs(best_trend):
                    best_trend = trend
                    best_sensor = name

        cycles = len(data.get("time_cycles", signals.get(list(signals.keys())[0]) if signals else [0]))
        if isinstance(cycles, np.ndarray):
            cycles = len(cycles)

        if best_sensor and best_trend != 0:
            rul_estimate = params.get("max_rul", 130) * (1 - abs(best_trend) / params.get("max_degradation_rate", 0.02))
            rul_estimate = max(0, min(params.get("max_rul", 130), rul_estimate))
            confidence_low = max(0, rul_estimate - 15)
            confidence_high = min(params.get("max_rul", 130), rul_estimate + 15)
        else:
            rul_estimate = params.get("max_rul", 130) / 2
            confidence_low = 0
            confidence_high = params.get("max_rul", 130)

        elapsed_ms = int((time.time() - t0) * 1000)
        return AssetRunResult(
            execution_status="success",
            validity_status="degraded",  # 线性趋势 ≠ CNN-LSTM
            can_influence_decision=False,
            structured_output={
                "rul_cycles": round(rul_estimate, 1),
                "confidence_interval": [round(confidence_low, 1), round(confidence_high, 1)],
                "method": "linear_trend_baseline",
                "sensor_used": best_sensor,
                "cycles_observed": cycles,
                "note": (
                    "LINEAR TREND BASELINE — NOT CNN-LSTM. "
                    "This implementation fits a linear trend per sensor and extrapolates RUL "
                    "using a fixed max degradation rate. "
                    "No operating condition normalization, no sensor covariance, no healthy baseline, "
                    "no confidence calibration. Confidence interval is fixed ±15 cycles. "
                    "CNN-LSTM requires: TensorFlow + trained weights on C-MAPSS (FD001-FD004) "
                    "with piecewise-linear RUL target function."
                ),
            },
            warnings=[
                "Asset ID 'cnn_lstm_cmapss' does NOT match implementation 'linear_trend_baseline'. "
                "C-MAPSS benchmark performance metrics in manifest do NOT apply to this baseline."
            ],
            metrics={"rul_mean": float(rul_estimate), "inference_ms": elapsed_ms},
            elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def default_params() -> dict:
        return {"max_rul": 130, "min_rul": 0, "max_degradation_rate": 0.02, "sequence_length": 50}
