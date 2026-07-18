"""信号检测器 — WCamba轴承故障 (CNN+Mamba) / FaultSense (LSTM自编码器)

完整模型自动加载 → pip install torch tensorflow 后自动切换到深度学习推理。
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from ._base import AssetRunResult, ImplementationBase

logger = logging.getLogger("aero_diag.detectors.signal")


def _resolve_repo_path(env_var: str, default_dir: str) -> Path | None:
    candidates = [
        Path(os.environ.get(env_var, "")),
        Path.cwd() / default_dir,
        Path.home() / default_dir,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ═══════════════════════════════════════════════════════════════════════
# WCamba 轴承故障诊断 — 完整：CNN+Mamba推理 | 基线：包络谱
# ═══════════════════════════════════════════════════════════════════════

class WCambaBearingFaultDetector(ImplementationBase):
    """WCamba 轻量 CNN+Mamba 轴承故障诊断。

    完整模型: git clone https://github.com/CDUT-IMRT/WCamba
    加载 PyTorch 模型 → 4 类轴承故障分类。

    基线: Hilbert 包络谱 + 故障特征频率匹配。
    """

    asset_id = "detector.vibration.wcamba_bearing_fault"

    BEARING_FAULTS = {
        "inner_race": {"freq_ratio": 5.43},
        "outer_race": {"freq_ratio": 3.57},
        "ball": {"freq_ratio": 2.35},
        "cage": {"freq_ratio": 0.38},
    }

    def __init__(self):
        self._model = None
        self._loaded = False
        self._error = ""

    def _load_model(self, params: dict) -> bool:
        """加载 WCamba 领域权重。拒绝随机初始化网络。

        审计修复 (AER-009): 权重缺失必须 unavailable，不能创建随机网络。
        """
        if self._loaded:
            return True

        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self._error = "pip install torch required"
            return False

        # 仅从 WCamba 仓库加载训练好的权重
        repo = _resolve_repo_path("WCAMBA_REPO_PATH", "WCamba")
        if repo:
            sys.path.insert(0, str(repo))
            try:
                from models.wcamba import WCambaModel
                self._model = WCambaModel(
                    in_channels=1, num_classes=4,
                    d_model=params.get("d_model", 64),
                )
                weights = repo / "checkpoints" / "best_model.pth"
                if weights.exists():
                    state = torch.load(str(weights), map_location="cpu")
                    self._model.load_state_dict(state)
                    self._model.eval()
                    self._loaded = True
                    self._model_type = "wcamba_cnn_mamba_trained"
                    return True
                else:
                    self._error = (
                        "WCamba repo found but no trained weights (checkpoints/best_model.pth). "
                        "Train on bearing fault dataset (CWRU/PU/XJTU-SY) first."
                    )
            except ImportError as e:
                self._error = f"WCamba repo found but import failed: {e}"
            except Exception as e:
                self._error = f"WCamba load failed: {e}"
            finally:
                if str(repo) in sys.path:
                    sys.path.remove(str(repo))
        else:
            self._error = (
                "WCamba model not found. "
                "git clone https://github.com/CDUT-IMRT/WCamba and train on bearing dataset. "
                "A randomly initialized CNN has NO statistical meaning for fault diagnosis."
            )

        return False

    def _envelope_baseline(self, sig: np.ndarray, fs: float, speed_rpm: float) -> dict:
        """Hilbert 包络谱 + 故障频率匹配（已有基线）。"""
        from scipy import signal, fft

        analytic = signal.hilbert(sig)
        envelope = np.abs(analytic)
        n = len(envelope)
        fft_len = min(4096, n)
        env_fft = np.abs(fft.rfft(envelope[:fft_len]))
        env_freq = fft.rfftfreq(fft_len, 1.0 / fs)

        shaft_freq = speed_rpm / 60.0
        fault_scores = {}
        for ftype, info in self.BEARING_FAULTS.items():
            target = shaft_freq * info["freq_ratio"]
            if target < fs / 2:
                idx = np.argmin(np.abs(env_freq - target))
                energy = float(env_fft[idx])
                for h in [2, 3]:
                    h_f = target * h
                    if h_f < fs / 2:
                        h_idx = np.argmin(np.abs(env_freq - h_f))
                        energy += float(env_fft[h_idx]) * 0.5
                fault_scores[ftype] = energy

        best = max(fault_scores, key=fault_scores.get) if fault_scores else "none"
        total_e = float(np.sum(env_fft))
        norm_score = fault_scores.get(best, 0) / max(total_e, 1e-10) if best != "none" else 0.0
        return {
            "fault_detected": bool(norm_score > 0.05 and best != "none"),
            "fault_type": best if norm_score > 0.05 else "none",
            "fault_scores": {k: round(float(v), 4) for k, v in fault_scores.items()},
            "shaft_frequency_hz": round(shaft_freq, 2),
        }

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        if not inputs:
            return {"ok": False, "issues": ["No vibration signal"]}
        data = inputs[0] if isinstance(inputs[0], dict) else {}
        if not data.get("vibration") and not data.get("signal"):
            return {"ok": False, "issues": ["vibration or signal field required"]}
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        params = {**self.default_params(), **parameters}
        data = inputs[0] if inputs else {}
        sig = np.asarray(data.get("vibration", data.get("signal", [])), dtype=np.float64).flatten()
        fs = float(data.get("sample_rate", params.get("sample_rate", 10000.0)))
        speed_rpm = float(data.get("speed_rpm", params.get("speed_rpm", 3000.0)))

        # 完整深度学习推理 — 仅当 WCamba 训练权重加载成功
        if self._load_model(params) and len(sig) >= 2048:
            try:
                import torch
                seg = sig[:2048]
                tensor = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    logits = self._model(tensor)
                    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
                class_names = ["inner_race", "outer_race", "ball", "cage"]
                pred_idx = int(np.argmax(probs))
                return AssetRunResult.valid_success(
                    output={
                        "fault_detected": bool(probs[pred_idx] > 0.5),
                        "fault_type": class_names[pred_idx],
                        "probabilities": {class_names[i]: round(float(probs[i]), 4) for i in range(4)},
                        "method": self._model_type,
                    },
                    model_identity=self._model_type,
                    metrics={"max_probability": float(probs[pred_idx]), "signal_rms": float(np.sqrt(np.mean(sig**2)))},
                )
            except Exception as e:
                return AssetRunResult(
                    execution_status="failed", validity_status="invalid",
                    error=f"WCamba inference failed: {e}",
                    structured_output={}, metrics={},
                )

        # 基线 — 包络谱分析（工程中有意义但不是 DL 故障诊断）
        result = self._envelope_baseline(sig, fs, speed_rpm)
        return AssetRunResult.degraded(
            method="envelope_spectrum_baseline",
            reason=(
                f"WCamba trained weights not loaded: {self._error}. "
                "Envelope spectrum baseline requires bearing geometry parameters "
                "(BPFO/BPFI/BSF/FTF) for accurate fault frequency identification. "
                "Fixed frequency ratios are approximations that may not match specific bearings."
            ),
            output=result,
            metrics={"signal_rms": float(np.sqrt(np.mean(sig**2)))},
        )

    @staticmethod
    def default_params() -> dict:
        return {"sample_rate": 10000.0, "speed_rpm": 3000.0, "detection_threshold": 0.05, "d_model": 64}


# ═══════════════════════════════════════════════════════════════════════
# FaultSense LSTM 自编码器 — 完整：TensorFlow LSTM-AE | 基线：趋势
# ═══════════════════════════════════════════════════════════════════════

class FaultSenseLSTMDetector(ImplementationBase):
    """FaultSense LSTM 自编码器异常检测。

    完整模型: git clone https://github.com/momo-2609/FaultSense-LSTM-Anomaly-Detection-on-NASA-CMAPSS
    加载 TensorFlow/Keras LSTM-AE → 重构误差异常评分。

    基线: 滑动窗口退化趋势分析。
    """

    asset_id = "detector.timeseries.faultsense_lstm_autoencoder"

    CMAPSS_SENSORS = [
        "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc",
        "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed",
        "Nf_dmd", "PCNfR_dmd", "W31", "W32",
    ]

    def __init__(self):
        self._model = None
        self._loaded = False
        self._error = ""

    def _load_model(self, params: dict) -> bool:
        """加载 FaultSense 训练好的 LSTM-AE。拒绝未训练模型。

        审计修复 (AER-009): 未训练的 LSTM-AE 重构误差没有可解释性。
        """
        if self._loaded:
            return True

        # 仅从 FaultSense 仓库加载已训练的 .h5 模型
        repo = _resolve_repo_path("FAULTSENSE_REPO_PATH", "FaultSense-LSTM-Anomaly-Detection-on-NASA-CMAPSS")
        if repo:
            try:
                import tensorflow as tf
                model_path = repo / "models" / "lstm_autoencoder.h5"
                if model_path.exists():
                    self._model = tf.keras.models.load_model(str(model_path))
                    self._loaded = True
                    self._model_type = "lstm_ae_trained"
                    return True
                else:
                    self._error = (
                        "FaultSense repo found but no trained model (models/lstm_autoencoder.h5). "
                        "Train on C-MAPSS dataset first: "
                        "git clone https://github.com/momo-2609/FaultSense-LSTM-Anomaly-Detection-on-NASA-CMAPSS"
                    )
            except ImportError:
                self._error = "pip install tensorflow required for FaultSense LSTM-AE"
            except Exception as e:
                self._error = f"TF model load failed: {e}"
        else:
            self._error = (
                "FaultSense LSTM-AE model not found. "
                "git clone https://github.com/momo-2609/FaultSense-LSTM-Anomaly-Detection-on-NASA-CMAPSS "
                "and train on C-MAPSS. "
                "An UNTRAINED LSTM autoencoder produces meaningless reconstruction errors."
            )

        return False

    def _trend_baseline(self, sensor_values: dict, params: dict) -> dict:
        """滑动窗口退化趋势基线。"""
        trends = [v.get("trend", 0) for v in sensor_values.values() if "trend" in v]
        if trends:
            max_abs = max(abs(t) for t in trends)
            score = min(max_abs / params.get("trend_threshold", 0.005), 1.0)
        else:
            score = 0.0
        is_anomalous = score > 0.25
        return {"anomaly_detected": bool(is_anomalous), "anomaly_score": round(float(score), 6)}

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        if not inputs:
            return {"ok": False, "issues": ["No sensor data"]}
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        params = {**self.default_params(), **parameters}
        data = inputs[0] if inputs else {}

        # 提取传感器统计
        sensor_values = {}
        sensor_arrays = {}
        for ch in self.CMAPSS_SENSORS:
            val = data.get(ch)
            if val is not None:
                if isinstance(val, (list, np.ndarray)):
                    arr = np.asarray(val, dtype=np.float64)
                    sensor_values[ch] = {
                        "mean": float(np.mean(arr)), "std": float(np.std(arr)),
                        "trend": float(np.polyfit(np.arange(len(arr)), arr, 1)[0]) if len(arr) > 5 else 0,
                    }
                    sensor_arrays[ch] = arr
                else:
                    sensor_values[ch] = {"value": float(val)}

        # 完整 LSTM-AE 推理 — 仅当训练好的模型加载
        if self._loaded and len(sensor_arrays) >= 3:
            try:
                seq_len = params.get("sequence_length", 50)
                arrays_list = []
                for ch in self.CMAPSS_SENSORS[:21]:
                    arr = sensor_arrays.get(ch, np.zeros(seq_len))
                    if len(arr) < seq_len:
                        arr = np.pad(arr, (0, seq_len - len(arr)))
                    arrays_list.append(arr[:seq_len])
                if len(arrays_list) >= 3:
                    X = np.stack(arrays_list, axis=-1).reshape(1, seq_len, -1)
                    X_pred = self._model.predict(X, verbose=0)
                    mse = float(np.mean((X - X_pred) ** 2))
                    score = min(mse / params.get("mse_threshold", 0.01), 1.0)
                    return AssetRunResult.valid_success(
                        output={
                            "anomaly_detected": bool(score > params.get("threshold_sigma", 2.5) / 10),
                            "anomaly_score": round(score, 6),
                            "reconstruction_mse": round(mse, 6),
                            "method": "lstm_autoencoder_trained",
                        },
                        model_identity="lstm_ae_trained",
                        metrics={"anomaly_score": score, "reconstruction_mse": mse, "sensor_count": float(len(sensor_arrays))},
                    )
            except Exception as e:
                return AssetRunResult(
                    execution_status="failed", validity_status="invalid",
                    error=f"LSTM-AE inference failed: {e}",
                    structured_output={}, metrics={},
                )

        # 基线 — 简单趋势分析，没有工况归一化和协变量处理
        result = self._trend_baseline(sensor_values, params)
        return AssetRunResult.degraded(
            method="sliding_window_trend_baseline",
            reason=(
                f"FaultSense trained LSTM-AE not loaded: {self._error}. "
                "Trend baseline uses linear slope with hardcoded thresholds — "
                "no operating condition normalization, no sensor covariance handling, "
                "no healthy baseline, no confidence intervals."
            ),
            output=result,
            metrics={"anomaly_score": result["anomaly_score"], "sensor_count": float(len(sensor_values))},
        )

    @staticmethod
    def default_params() -> dict:
        return {"sequence_length": 50, "n_features": 21, "threshold_sigma": 2.5, "trend_threshold": 0.005, "mse_threshold": 0.01}
