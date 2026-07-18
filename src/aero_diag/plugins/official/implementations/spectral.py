"""SciPy 频谱分析预处理器 — P0_CRITICAL, 零额外依赖"""

import numpy as np
from scipy import signal, fft

from ._base import AssetRunResult, ImplementationBase


class SpectralAnalysis(ImplementationBase):
    """SciPy 频谱分析预处理器。

    提供 FFT / 功率谱密度(PSD) / STFT(时频谱) / 包络谱 / 倒频谱。
    输入为 numpy 数组（振动信号 + 采样率），输出为结构化频谱数据。
    """

    asset_id = "preprocessor.signal.scipy_spectral_analysis"

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        """验证输入：至少需要 signal 和 sample_rate。"""
        if not inputs:
            return {"ok": False, "issues": ["No input signal provided"]}
        data = inputs[0] if isinstance(inputs[0], dict) else {}
        params = {**self.default_params(), **parameters}
        if "sample_rate" not in data and "sample_rate" not in params:
            return {"ok": False, "issues": ["sample_rate is required"]}
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        params = {**self.default_params(), **parameters}
        data = inputs[0] if inputs else {}

        # 获取信号
        sig = data.get("signal", data.get("data", data.get("vibration")))
        if sig is None:
            return AssetRunResult(
                status="failed", structured_output={},
                warnings=["No signal found in input"], metrics={},
            )
        sig = np.asarray(sig, dtype=np.float64).flatten()
        fs = float(data.get("sample_rate", params.get("sample_rate", 1.0)))
        n = len(sig)

        fft_len = int(params.get("fft_length", 4096))
        fft_len = min(fft_len, n)  # 实际使用不超信号长度
        results = {}

        # FFT 幅值谱
        if params.get("compute_fft", True):
            freq = fft.rfftfreq(fft_len, 1.0 / fs)
            mag = np.abs(fft.rfft(sig[:fft_len] * signal.windows.hann(fft_len)))
            results["fft"] = {
                "frequencies_hz": freq[:fft_len // 2].tolist(),
                "magnitudes": mag[:fft_len // 2].tolist(),
                "dominant_freqs": freq[np.argsort(mag)[-6:]].tolist(),
            }

        # PSD
        if params.get("compute_psd", True):
            f_pxx, pxx = signal.welch(
                sig, fs, nperseg=min(fft_len, n),
                noverlap=min(fft_len // 2, n // 2),
            )
            results["psd"] = {
                "frequencies_hz": f_pxx.tolist(),
                "power": pxx.tolist(),
            }

        # STFT
        if params.get("compute_stft", True) and n > fft_len:
            f_stft, t_stft, Zxx = signal.spectrogram(
                sig, fs, nperseg=min(fft_len, n // 4),
                noverlap=min(fft_len * 3 // 4, n // 5),
            )
            results["stft"] = {
                "frequencies_hz": f_stft.tolist(),
                "times_s": t_stft.tolist(),
                "spectrogram": Zxx.tolist(),
            }

        # 包络谱 (Hilbert)
        if params.get("compute_envelope", False):
            analytic = signal.hilbert(sig)
            envelope = np.abs(analytic)
            env_fft = np.abs(fft.rfft(envelope[:fft_len]))
            env_freq = fft.rfftfreq(fft_len, 1.0 / fs)
            results["envelope"] = {
                "frequencies_hz": env_freq[:200].tolist(),
                "magnitudes": env_fft[:200].tolist(),
                "dominant_freqs": env_freq[np.argsort(env_fft[:200])[-6:]].tolist(),
            }

        return AssetRunResult(
            status="success",
            structured_output={
                "signal_length": n,
                "sample_rate_hz": fs,
                "duration_s": n / fs,
                "analysis": results,
            },
            metrics={
                "rms": float(np.sqrt(np.mean(sig**2))),
                "peak": float(np.max(np.abs(sig))),
                "crest_factor": float(np.max(np.abs(sig)) / np.sqrt(np.mean(sig**2))) if np.sqrt(np.mean(sig**2)) > 0 else 0,
            },
        )

    @staticmethod
    def default_params() -> dict:
        return {
            "fft_length": 4096,
            "compute_fft": True,
            "compute_psd": True,
            "compute_stft": True,
            "compute_envelope": False,
        }
