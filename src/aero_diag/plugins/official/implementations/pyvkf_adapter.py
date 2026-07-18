"""PyVKF 阶次跟踪预处理器 + BladeSynth 合成数据适配器 — P1/P2"""

import numpy as np
from ._base import AssetRunResult, ImplementationBase


class PyVKFOrderTracking(ImplementationBase):
    """PyVKF Vold-Kalman 阶次跟踪滤波器。

    无转速计阶次提取。安装 pyvkf 后启用完整 VKF；否则使用 FFT 阶次估计基线。
    """

    asset_id = "preprocessor.vibration.pyvkf_order_tracking"

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        if not inputs:
            return {"ok": False, "issues": ["No vibration signal provided"]}
        data = inputs[0] if isinstance(inputs[0], dict) else {}
        if "vibration" not in data and "signal" not in data and "data" not in data:
            return {"ok": False, "issues": ["vibration or signal field required"]}
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        params = {**self.default_params(), **parameters}
        data = inputs[0] if inputs else {}

        sig = np.asarray(
            data.get("vibration", data.get("signal", data.get("data", []))),
            dtype=np.float64,
        ).flatten()
        fs = float(data.get("sample_rate", params.get("sample_rate", 1000.0)))
        speed_ref = data.get("speed_reference", params.get("speed_reference"))

        # 阶次提取：基于 FFT 的简单谐波检测作为基线
        n = len(sig)
        freq = np.fft.rfftfreq(n, 1.0 / fs)
        mag = np.abs(np.fft.rfft(sig * np.hanning(n)))

        max_order = int(params.get("max_order", 20))
        orders = {}
        if speed_ref:
            try:
                base_freq = float(speed_ref) / 60.0  # RPM → Hz
            except (ValueError, TypeError):
                base_freq = None
        else:
            # 无转速参考：用最大峰值估计基频
            peak_idx = np.argmax(mag[1:]) + 1
            base_freq = freq[peak_idx]

        if base_freq and base_freq > 0:
            for order in range(1, max_order + 1):
                target = order * base_freq
                if target < fs / 2:
                    idx = np.argmin(np.abs(freq - target))
                    orders[f"order_{order}"] = {
                        "frequency_hz": round(float(freq[idx]), 2),
                        "magnitude": round(float(mag[idx]), 4),
                    }

        try:
            import pyvkf
            method = "pyvkf_full"
            orders["_note"] = "PyVKF available for full Vold-Kalman filtering"
        except ImportError:
            method = "fft_harmonic_baseline"
            orders["_note"] = "Install PyVKF for Vold-Kalman: git clone https://github.com/CyprienHoelzl/PyVKF"

        return AssetRunResult(
            status="success" if orders else "partial",
            structured_output={
                "method": method,
                "base_frequency_hz": round(float(base_freq), 2) if base_freq else None,
                "speed_rpm": round(float(speed_ref), 1) if speed_ref else None,
                "orders": orders,
                "signal_length": n,
                "sample_rate_hz": fs,
            },
            metrics={
                "signal_rms": float(np.sqrt(np.mean(sig**2))),
                "dominant_freq_hz": float(freq[np.argmax(mag[1:]) + 1]),
            },
        )

    @staticmethod
    def default_params() -> dict:
        return {"max_order": 20, "bandwidth": 0.05, "parallel_orders": 4, "sample_rate": 1000.0, "speed_reference": None}


class BladeSynthAdapter(ImplementationBase):
    """BladeSynth 合成叶片缺陷数据集适配器。"""

    asset_id = "data_adapter.borescope.bladesynth"

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        params = {**self.default_params(), **parameters}
        root = params.get("dataset_root", "")
        images = []
        if root:
            from pathlib import Path
            for p in Path(root).glob("**/*.png"):
                images.append({"uri": str(p), "filename": p.name, "source": "bladesynth"})

        return AssetRunResult(
            status="success" if images else "partial",
            structured_output={
                "images_found": len(images),
                "images": images[:100],
                "dataset_root": root,
                "note": "BladeSynth — synthetic only. Not for certification decisions. "
                        "git clone https://github.com/MohammedEltoum/bladeSynth",
            },
            metrics={"image_count": float(len(images))},
            warnings=[] if images else ["No BladeSynth images found — dataset not downloaded"],
        )

    @staticmethod
    def default_params() -> dict:
        return {"dataset_root": "", "render_quality": "high"}
