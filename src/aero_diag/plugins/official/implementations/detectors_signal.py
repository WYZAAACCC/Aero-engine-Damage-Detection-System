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
    """WCamba 轴承故障诊断 — 4类 (normal/inner_race/outer_race/ball)。

    优先加载训练权重 (artifacts/models/wcamba_cwru_4class/),
    其次尝试上游 WCamba 仓库, 否则使用包络谱基线。
    """

    asset_id = "detector.vibration.wcamba_bearing_fault"

    CLASS_NAMES = ["normal", "inner_race", "outer_race", "ball"]

    BEARING_FAULTS = {
        "inner_race": {"freq_ratio": 5.43},
        "outer_race": {"freq_ratio": 3.57},
        "ball": {"freq_ratio": 2.35},
    }

    # ── 训练权重搜索路径 ──
    TRAINED_WEIGHT_PATHS = [
        Path("artifacts/models/wcamba_cwru_4class/v1.0/best_model.pth"),
        Path("artifacts/models/wcamba_cwru_4class/v1.0/best_model.pth").absolute(),
    ]

    def __init__(self):
        self._model = None
        self._loaded = False
        self._error = ""
        self._model_type = "none"

    def _build_model(self):
        """构建与训练时一致的 WideKernel 1D-CNN。"""
        import torch.nn as nn

        class WideKernel1DCNN(nn.Module):
            def __init__(self, in_channels=1, num_classes=4, d_model=64):
                super().__init__()
                self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=64, stride=2, padding=32)
                self.bn1 = nn.BatchNorm1d(16)
                self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
                self.bn2 = nn.BatchNorm1d(32)
                self.conv3 = nn.Conv1d(32, d_model, kernel_size=3, stride=2, padding=1)
                self.bn3 = nn.BatchNorm1d(d_model)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(d_model, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)

            def forward(self, x):
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.relu(self.bn2(self.conv2(x)))
                x = self.relu(self.bn3(self.conv3(x)))
                x = self.pool(x).squeeze(-1)
                x = self.dropout(x)
                return self.fc(x)

        return WideKernel1DCNN(in_channels=1, num_classes=4)

    def _load_model(self, params: dict) -> bool:
        """加载训练权重。按优先级:
        1. 本仓库训练产物 (artifacts/models/wcamba_cwru_4class/)
        2. 上游 WCamba 仓库权重
        """
        if self._loaded:
            return True

        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self._error = "pip install torch required"
            return False

        # ── 方案A: 加载本仓库训练权重 ──
        for p in self.TRAINED_WEIGHT_PATHS:
            if p.exists():
                try:
                    self._model = self._build_model()
                    checkpoint = torch.load(str(p), map_location="cpu", weights_only=False)
                    self._model.load_state_dict(checkpoint["model_state_dict"])
                    self._model.eval()
                    self._loaded = True
                    self._model_type = "wcamba_cwru_4class_trained"
                    # 验证权重
                    class_names = checkpoint.get("class_names", self.CLASS_NAMES)
                    self.CLASS_NAMES = class_names
                    val_acc = checkpoint.get("val_accuracy", 0)
                    print(f"[WCamba] Loaded trained weights from {p} "
                          f"(val_acc={val_acc:.4f}, classes={class_names})")
                    return True
                except Exception as e:
                    self._error = f"Failed to load trained weights from {p}: {e}"
                    self._model = None

        # ── 方案B: 上游 WCamba 仓库 ──
        repo = _resolve_repo_path("WCAMBA_REPO_PATH", "WCamba")
        if repo:
            sys.path.insert(0, str(repo))
            try:
                from models.wcamba import WCambaModel  # noqa: F811
                self._model = WCambaModel(in_channels=1, num_classes=4,
                                          d_model=params.get("d_model", 64))
                weights = repo / "checkpoints" / "best_model.pth"
                if weights.exists():
                    state = torch.load(str(weights), map_location="cpu", weights_only=False)
                    self._model.load_state_dict(state)
                    self._model.eval()
                    self._loaded = True
                    self._model_type = "wcamba_cnn_mamba_upstream"
                    return True
                else:
                    self._error = "WCamba repo found but no checkpoints/best_model.pth"
            except ImportError as e:
                self._error = f"WCamba repo import failed: {e}"
            except Exception as e:
                self._error = f"WCamba load failed: {e}"
            finally:
                if str(repo) in sys.path:
                    sys.path.remove(str(repo))

        if not self._error:
            self._error = (
                "No trained weights found. Train WCamba on CWRU data first: "
                "python training/scripts/train/train_wcamba.py --data artifacts/raw/cwru"
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
        # 输入窗口1024点（与训练时一致），per-window z-score归一化
        if self._load_model(params) and len(sig) >= 1024:
            try:
                import torch
                seg = sig[:1024].astype(np.float32)
                # per-window z-score 归一化（与训练时一致）
                std = seg.std()
                if std > 1e-8:
                    seg = (seg - seg.mean()) / std
                tensor = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    logits = self._model(tensor)
                    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
                class_names = self.CLASS_NAMES
                pred_idx = int(np.argmax(probs))
                return AssetRunResult.valid_success(
                    output={
                        "fault_detected": bool(pred_idx != 0 or probs[pred_idx] < 0.9),
                        "fault_type": class_names[pred_idx],
                        "probabilities": {class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))},
                        "method": self._model_type,
                        "model_version": "1.0",
                    },
                    model_identity=self._model_type,
                    metrics={"max_probability": float(probs[pred_idx]),
                             "signal_rms": float(np.sqrt(np.mean(sig**2)))},
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
    """FaultSense LSTM-AE 异常检测 + RUL — PyTorch移植版。

    加载本仓库训练权重 (artifacts/models/faultsense/),
    否则回退滑动窗口趋势基线。
    """

    asset_id = "detector.timeseries.faultsense_lstm_autoencoder"

    TRAINED_WEIGHT_PATHS = [
        Path("artifacts/models/faultsense/v1.0/FD001/best_model.pth"),
        Path("artifacts/models/faultsense/v1.0/FD003/best_model.pth"),
    ]
    SEQ_LEN = 30
    N_FEATURES = 14
    RUL_CAP = 130

    def __init__(self):
        self._model = None
        self._loaded = False
        self._error = ""
        self._config = {}
        self._model_identity = "none"

    def _build_model(self):
        """构建 FaultSense LSTM-AE + RUL (与训练一致)。"""
        import torch
        import torch.nn as nn

        class FaultSenseModel(nn.Module):
            def __init__(self, n_features=14, hidden=32, dropout=0.5):
                super().__init__()
                self.encoder = nn.LSTM(n_features, hidden, num_layers=2,
                                       batch_first=True, dropout=dropout)
                self.decoder = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
                self.reconstruct = nn.Linear(hidden, n_features)
                self.rul_fc = nn.Sequential(
                    nn.Linear(hidden, hidden // 2), nn.ReLU(),
                    nn.Dropout(dropout), nn.Linear(hidden // 2, 1), nn.Sigmoid())
                t = torch.tensor(0.0)
                self.register_buffer('threshold', t)

            def encode(self, x):
                _, (h_n, _) = self.encoder(x)
                return h_n[-1]

            def decode(self, h, seq_len):
                h = h.unsqueeze(1).repeat(1, seq_len, 1)
                dec_out, _ = self.decoder(h)
                return self.reconstruct(dec_out)

            def forward(self, x):
                seq_len = x.size(1)
                h = self.encode(x)
                recon = self.decode(h, seq_len)
                rul_pred = self.rul_fc(h)
                return recon, rul_pred

        return FaultSenseModel(n_features=self.N_FEATURES)

    def _load_model(self) -> bool:
        """加载训练好的 FaultSense PyTorch 权重。"""
        if self._loaded:
            return True
        try:
            import torch
        except ImportError:
            self._error = "pip install torch required"
            return False

        for p in self.TRAINED_WEIGHT_PATHS:
            if p.exists():
                try:
                    self._model = self._build_model()
                    checkpoint = torch.load(str(p), map_location="cpu", weights_only=False)
                    self._model.load_state_dict(checkpoint["model_state_dict"])
                    self._model.eval()
                    self._loaded = True
                    self._config = {
                        "subset": checkpoint.get("subset", "FD001"),
                        "threshold": float(checkpoint.get("threshold", 0.5)),
                        "val_rmse": checkpoint.get("val_rmse", 0),
                    }
                    self._model_identity = f"faultsense_lstm_ae_{self._config['subset'].lower()}_trained"
                    print(f"[FaultSense] Loaded trained weights from {p} "
                          f"(subset={self._config['subset']}, val_rmse={self._config['val_rmse']:.1f})")
                    return True
                except Exception as e:
                    self._error = f"Failed to load {p}: {e}"
                    self._model = None

        if not self._error:
            self._error = (
                "No trained FaultSense weights found. Train first: "
                "python training/scripts/train/train_faultsense.py --subset FD001"
            )
        return False

    def _trend_baseline(self, signals: dict) -> dict:
        """滑动窗口退化趋势基线。"""
        trends = []
        for arr in signals.values():
            if isinstance(arr, np.ndarray) and len(arr) > 10:
                t = np.polyfit(np.arange(len(arr)), arr, 1)[0]
                trends.append(t)
        max_abs = max(abs(t) for t in trends) if trends else 0
        score = min(max_abs / 0.005, 1.0)
        return {"anomaly_detected": score > 0.25, "anomaly_score": round(float(score), 6)}

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        if not inputs:
            return {"ok": False, "issues": ["No sensor data"]}
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        data = inputs[0] if inputs else {}

        # ── 真实 LSTM-AE 推理 ──
        if self._load_model():
            try:
                import torch
                # 提取传感器序列
                cmapss_keys = [f"s{i}" for i in
                    [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
                arrays = []
                for k in cmapss_keys:
                    v = data.get(k)
                    if isinstance(v, (list, np.ndarray)):
                        arrays.append(np.asarray(v, dtype=np.float32))
                # 回退: 使用任何可用序列
                if len(arrays) < 3:
                    for v in data.values():
                        if isinstance(v, (list, np.ndarray)) and len(v) >= self.SEQ_LEN:
                            arrays.append(np.asarray(v, dtype=np.float32))
                        if len(arrays) >= self.N_FEATURES:
                            break
                if len(arrays) >= 3:
                    # 补齐并对齐
                    while len(arrays) < self.N_FEATURES:
                        arrays.append(np.zeros_like(arrays[0]))
                    min_len = min(len(a) for a in arrays)
                    seq = np.stack([a[-min_len:] for a in arrays[:self.N_FEATURES]], axis=-1)
                    if seq.shape[0] < self.SEQ_LEN:
                        seq = np.pad(seq, ((self.SEQ_LEN - seq.shape[0], 0), (0, 0)), mode='edge')
                    seq = seq[-self.SEQ_LEN:].astype(np.float32)

                    x = torch.tensor(seq).float().unsqueeze(0)
                    with torch.no_grad():
                        recon, rul_pred = self._model(x)
                        mse = float(torch.mean((x - recon) ** 2))
                    threshold = self._config.get("threshold", 0.6)
                    is_anomaly = mse > threshold
                    rul = max(0, float(rul_pred.item() * self.RUL_CAP))

                    return AssetRunResult.valid_success(
                        output={
                            "anomaly_detected": is_anomaly,
                            "anomaly_score": round(mse, 6),
                            "reconstruction_mse": round(mse, 6),
                            "threshold": round(threshold, 6),
                            "rul_cycles": round(rul, 1),
                            "method": self._model_identity,
                            "subset": self._config.get("subset", "unknown"),
                        },
                        model_identity=self._model_identity,
                        metrics={"anomaly_score": mse, "reconstruction_mse": mse, "rul": rul},
                    )
            except Exception as e:
                return AssetRunResult(
                    execution_status="failed", validity_status="invalid",
                    error=f"FaultSense inference failed: {e}",
                    structured_output={}, metrics={},
                )

        # ── 基线 ──
        signals = {k: np.asarray(v, dtype=np.float64) for k, v in data.items()
                    if isinstance(v, (list, np.ndarray))}
        result = self._trend_baseline(signals)
        return AssetRunResult.degraded(
            method="sliding_window_trend_baseline",
            reason=f"FaultSense trained weights not loaded: {self._error}",
            output=result,
            metrics={"anomaly_score": result["anomaly_score"]},
        )

    @staticmethod
    def default_params() -> dict:
        return {"sequence_length": 30, "n_features": 14, "threshold_sigma": 2.5, "trend_threshold": 0.005}
