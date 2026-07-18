"""CNN-LSTM RUL 预测器 — 训练权重 + 线性趋势基线。

审计修复 (AER-001): 资产 ID 与实现匹配。
优先加载训练权重 → 真实 CNN-LSTM 推理;
否则回退线性趋势基线 (degraded)。
"""

from pathlib import Path
import numpy as np

from ._base import AssetRunResult, ImplementationBase


class CNNLSTMRULPredictor(ImplementationBase):
    """CNN-LSTM RUL 预测器。

    优先: artifacts/models/cnn_lstm_rul/ 训练权重
    回退: 线性趋势基线 (validity=degraded)
    """

    asset_id = "reliability_model.rul.cnn_lstm_cmapss"

    # ── 训练权重路径 ──
    TRAINED_WEIGHT_PATHS = [
        Path("artifacts/models/cnn_lstm_rul/v1.0/FD001/best_model.pth"),
        Path("artifacts/models/cnn_lstm_rul/v1.0/FD003/best_model.pth"),
    ]

    # C-MAPSS 传感器索引 (训练时使用的14个传感器)
    CMAPSS_SENSOR_INDICES = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]  # 0-based
    N_FEATURES = 14
    SEQ_LEN = 50
    RUL_CAP = 130

    def __init__(self):
        self._model = None
        self._loaded = False
        self._error = ""
        self._config = {}
        self._model_identity = "none"

    def _build_model(self):
        """构建与训练时一致的 CNN-LSTM 模型。"""
        import torch.nn as nn

        class CNNLSTMRUL(nn.Module):
            def __init__(self, n_features=14, seq_len=50,
                         conv_channels=32, lstm_hidden=64, dropout=0.2):
                super().__init__()
                self.conv1 = nn.Conv1d(n_features, conv_channels, kernel_size=5, padding=2)
                self.bn1 = nn.BatchNorm1d(conv_channels)
                self.conv2 = nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm1d(conv_channels * 2)
                self.lstm = nn.LSTM(conv_channels * 2, lstm_hidden, num_layers=2,
                                    batch_first=True, dropout=dropout, bidirectional=True)
                self.fc = nn.Sequential(
                    nn.Linear(lstm_hidden * 2, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                x = x.transpose(1, 2)
                x = nn.functional.relu(self.bn1(self.conv1(x)))
                x = nn.functional.relu(self.bn2(self.conv2(x)))
                x = x.transpose(1, 2)
                lstm_out, _ = self.lstm(x)
                x = lstm_out[:, -1, :]
                return self.fc(x)

        return CNNLSTMRUL(n_features=self.N_FEATURES, seq_len=self.SEQ_LEN)

    def _load_model(self) -> bool:
        """加载训练好的 CNN-LSTM 权重。"""
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
                    subset = checkpoint.get("subset", p.parent.name)
                    val_rmse = checkpoint.get("val_rmse", 0)
                    self._config = {
                        "subset": subset,
                        "seq_len": checkpoint.get("seq_len", self.SEQ_LEN),
                        "rul_cap": checkpoint.get("rul_cap", self.RUL_CAP),
                        "n_features": checkpoint.get("n_features", self.N_FEATURES),
                        "sensor_indices": checkpoint.get("sensor_indices", self.CMAPSS_SENSOR_INDICES),
                        "val_rmse": val_rmse,
                    }
                    self.SEQ_LEN = self._config["seq_len"]
                    self.RUL_CAP = self._config["rul_cap"]
                    self.N_FEATURES = self._config["n_features"]
                    self._model_identity = f"cnn_lstm_{subset.lower()}_trained"
                    print(f"[CNN-LSTM RUL] Loaded trained weights from {p} "
                          f"(subset={subset}, val_rmse={val_rmse:.1f})")
                    return True
                except Exception as e:
                    self._error = f"Failed to load {p}: {e}"
                    self._model = None

        if not self._error:
            self._error = (
                "No trained CNN-LSTM weights found. Train first: "
                "python training/scripts/train/train_cnn_lstm_rul.py --subset FD001"
            )
        return False

    def _extract_sequence(self, data: dict) -> np.ndarray | None:
        """从输入数据提取传感器序列。

        尝试按 C-MAPSS 顺序 (s2, s3, s4, s7, s8, s9...),
        如果输入使用物理传感器名称，则按可用顺序取前14个。
        """
        # 方案 A: 输入为 C-MAPSS 匿名传感器
        cmapss_keys = [f"s{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
        arrays = []
        for k in cmapss_keys:
            v = data.get(k)
            if v is not None and isinstance(v, (list, np.ndarray)):
                arrays.append(np.asarray(v, dtype=np.float32))
        if len(arrays) >= self.N_FEATURES:
            arrays = arrays[:self.N_FEATURES]
        else:
            # 方案 B: 使用任何可用的序列数据
            arrays = []
            for k, v in sorted(data.items()):
                if isinstance(v, (list, np.ndarray)) and len(v) >= self.SEQ_LEN:
                    arrays.append(np.asarray(v, dtype=np.float32))
                if len(arrays) >= self.N_FEATURES:
                    break

        if len(arrays) < 3:
            return None

        # 补齐到 N_FEATURES
        while len(arrays) < self.N_FEATURES:
            arrays.append(np.zeros(max(len(a) for a in arrays), dtype=np.float32))

        # 对齐长度，取最后 SEQ_LEN 步
        min_len = min(len(a) for a in arrays)
        seq = np.stack([a[-min_len:] for a in arrays], axis=-1)  # (time, features)
        if seq.shape[0] < self.SEQ_LEN:
            seq = np.pad(seq, ((self.SEQ_LEN - seq.shape[0], 0), (0, 0)), mode="edge")
        seq = seq[-self.SEQ_LEN:]
        return seq.astype(np.float32)

    def _linear_baseline(self, signals: dict, params: dict) -> dict:
        """线性趋势基线——当CNN-LSTM权重不可用时的回退。"""
        best_sensor, best_trend = None, 0
        for name, arr in signals.items():
            if isinstance(arr, (list, np.ndarray)) and len(arr) > 10:
                arr = np.asarray(arr, dtype=np.float64)
                trend = np.polyfit(np.arange(len(arr)), arr, 1)[0]
                if abs(trend) > abs(best_trend):
                    best_trend = trend
                    best_sensor = name

        if best_sensor and best_trend != 0:
            max_rul = params.get("max_rul", 130)
            max_deg = params.get("max_degradation_rate", 0.02)
            rul_est = max_rul * (1 - abs(best_trend) / max_deg)
            rul_est = max(0, min(max_rul, rul_est))
        else:
            rul_est = params.get("max_rul", 130) / 2

        return {
            "rul_cycles": round(rul_est, 1),
            "confidence_interval": [max(0, rul_est - 15), min(130, rul_est + 15)],
            "method": "linear_trend_baseline",
            "sensor_used": best_sensor,
        }

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        if not inputs:
            return {"ok": False, "issues": ["No sensor time series provided"]}
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        params = {**self.default_params(), **parameters}
        data = inputs[0] if inputs else {}
        import time
        t0 = time.time()

        # ── 真实 CNN-LSTM 推理 ──
        if self._load_model():
            seq = self._extract_sequence(data)
            if seq is not None:
                try:
                    import torch
                    tensor = torch.tensor(seq).float().unsqueeze(0)
                    with torch.no_grad():
                        pred = self._model(tensor).item() * self.RUL_CAP
                    rul = max(0, round(pred, 1))
                    elapsed_ms = int((time.time() - t0) * 1000)
                    return AssetRunResult.valid_success(
                        output={
                            "rul_cycles": rul,
                            "confidence_interval": [max(0, rul - 15), rul + 15],
                            "method": self._model_identity,
                            "subset": self._config.get("subset", "unknown"),
                            "seq_len": self.SEQ_LEN,
                            "rul_cap": self.RUL_CAP,
                            "note": (
                                "CNN-LSTM RUL prediction on C-MAPSS domain. "
                                "C-MAPSS is SIMULATED data — NOT real engine telemetry. "
                                "Sensors are anonymous (s1-s21) — do NOT map to physical quantities. "
                                f"Val RMSE={self._config.get('val_rmse', 'N/A')} cycles on {self._config.get('subset')}."
                            ),
                        },
                        model_identity=self._model_identity,
                        metrics={
                            "rul_cycles": float(rul),
                            "inference_ms": elapsed_ms,
                        },
                    )
                except Exception as e:
                    self._error = f"CNN-LSTM inference failed: {e}"

        # ── 线性趋势基线 ──
        signals = {}
        for key, val in data.items():
            if isinstance(val, (list, np.ndarray)):
                signals[key] = np.asarray(val, dtype=np.float64)

        result = self._linear_baseline(signals, params)
        elapsed_ms = int((time.time() - t0) * 1000)
        return AssetRunResult(
            execution_status="success",
            validity_status="degraded",
            can_influence_decision=False,
            structured_output={
                **result,
                "note": (
                    "LINEAR TREND BASELINE — NOT CNN-LSTM. "
                    f"CNN-LSTM weights not loaded: {self._error}. "
                    "This implementation fits a linear trend per sensor and extrapolates RUL "
                    "using a fixed max degradation rate."
                ),
            },
            warnings=[
                "Asset ID 'cnn_lstm_cmapss': using linear_trend_baseline fallback. "
                "Train CNN-LSTM on C-MAPSS for real RUL prediction."
            ],
            metrics={"rul_mean": float(result["rul_cycles"]), "inference_ms": elapsed_ms},
            elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def default_params() -> dict:
        return {"max_rul": 130, "min_rul": 0, "max_degradation_rate": 0.02, "sequence_length": 50}
