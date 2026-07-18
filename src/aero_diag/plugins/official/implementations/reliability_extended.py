"""扩展可靠性模型 — FDPP概率裂纹/pyLife S-N/PINN/ChangePoint-LSTM"""

import numpy as np
from ._base import AssetRunResult, ImplementationBase


class FDPPProbabilisticCrackGrowth(ImplementationBase):
    """FrameworkFDPP 概率裂纹扩展——Monte Carlo Paris Law。"""

    asset_id = "reliability_model.crack.framework_fdpp_probabilistic"

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        params = {**self.default_params(), **parameters}
        if not params.get("n_mc_samples"):
            return {"ok": False, "issues": ["n_mc_samples required"]}
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        import time
        t0 = time.time()
        params = {**self.default_params(), **parameters}
        data = inputs[0] if inputs else {}

        n_samples = int(params.get("n_mc_samples", 1000))
        C_log_mean = float(data.get("c_log_mean", params.get("c_log_mean", -28.5)))
        C_log_std = float(data.get("c_log_std", params.get("c_log_std", 0.15)))
        m_mean = float(data.get("m_mean", params.get("m_mean", 3.5)))
        m_std = float(data.get("m_std", params.get("m_std", 0.1)))
        a0_mean = float(data.get("initial_crack_mean_mm", params.get("initial_crack_mean_mm", 0.5)))
        a0_cov = float(data.get("initial_crack_cov", params.get("initial_crack_cov", 0.2)))

        rng = np.random.RandomState(params.get("random_state", 42))

        # 采样参数
        C_samples = 10 ** rng.normal(C_log_mean, C_log_std, n_samples)
        m_samples = rng.normal(m_mean, m_std, n_samples)
        a0_samples = np.abs(rng.normal(a0_mean, a0_mean * a0_cov, n_samples))
        ac = float(data.get("critical_crack_mm", params.get("critical_crack_mm", 10.0)))
        delta_sigma = float(data.get("delta_sigma_mpa", params.get("delta_sigma_mpa", 200.0)))

        # MC模拟
        n_fast = min(n_samples, 500)
        cycles_array = np.zeros(n_fast)
        geometry_factor = 1.12
        for i in range(n_fast):
            a = a0_samples[i] * 1e-3
            c = 0
            C_val = C_samples[i]
            m_val = max(2.0, m_samples[i])
            while a < ac * 1e-3 and c < 1e9:
                dK = delta_sigma * geometry_factor * np.sqrt(np.pi * a)
                da_dN = C_val * (dK ** m_val)
                da = min((ac * 1e-3 - a) / 100, (ac * 1e-3 - a))
                dN = da / max(da_dN, 1e-20)
                a += da
                c += int(max(dN, 1))
            cycles_array[i] = c

        cycles_array = cycles_array[cycles_array > 0]
        elapsed_ms = int((time.time() - t0) * 1000)

        if len(cycles_array) > 0:
            return AssetRunResult(
                status="success",
                structured_output={
                    "method": "monte_carlo_paris_law",
                    "n_samples": n_fast,
                    "rul_mean_cycles": float(np.mean(cycles_array)),
                    "rul_median_cycles": float(np.median(cycles_array)),
                    "rul_p2_5_cycles": float(np.percentile(cycles_array, 2.5)),
                    "rul_p97_5_cycles": float(np.percentile(cycles_array, 97.5)),
                    "parameters": {"C_log_mean": C_log_mean, "m_mean": m_mean},
                },
                metrics={
                    "rul_mean": float(np.mean(cycles_array)),
                    "rul_std": float(np.std(cycles_array)),
                    "pof_at_1e5": float(np.mean(cycles_array < 1e5)),
                },
                elapsed_ms=elapsed_ms,
            )
        return AssetRunResult(status="failed", error="No valid MC samples", structured_output={}, metrics={})

    @staticmethod
    def default_params() -> dict:
        return {"n_mc_samples": 1000, "random_state": 42, "c_log_mean": -28.5, "c_log_std": 0.15,
                "m_mean": 3.5, "m_std": 0.1, "initial_crack_mean_mm": 0.5, "initial_crack_cov": 0.2,
                "critical_crack_mm": 10.0, "delta_sigma_mpa": 200.0}


class PyLifeSNCurveCalculator(ImplementationBase):
    """pyLife S-N 曲线疲劳寿命——回退到 Basquin 公式。"""

    asset_id = "reliability_model.fatigue.pylife_sn_woehler"

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        params = {**self.default_params(), **parameters}
        data = inputs[0] if inputs else {}

        sigma_a = float(data.get("stress_amplitude_mpa", params.get("stress_amplitude_mpa", 300.0)))
        Sf = float(data.get("fatigue_strength_coeff_mpa", params.get("Sf", 900.0)))
        b = float(data.get("fatigue_strength_exponent", params.get("b", -0.1)))

        # Basquin: sigma_a = Sf * (2Nf)^b → Nf = 0.5 * (sigma_a / Sf)^(1/b)
        try:
            Nf = 0.5 * (sigma_a / Sf) ** (1.0 / b)
            Nf = max(1, min(Nf, 1e9))
            method = "basquin_equation"
        except Exception:
            Nf = 1e6
            method = "basquin_default"

        return AssetRunResult(
            status="success",
            structured_output={
                "cycles_to_failure": round(Nf, 0),
                "stress_amplitude_mpa": sigma_a,
                "method": method,
                "note": "Basquin equation baseline. Install pyLife for FKM nonlinear: pip install pylife",
            },
            metrics={"cycles_to_failure": float(Nf), "stress_amplitude_mpa": sigma_a},
        )

    @staticmethod
    def default_params() -> dict:
        return {"stress_amplitude_mpa": 300.0, "Sf": 900.0, "b": -0.1}


class PinnFleetPrognosis(ImplementationBase):
    """PINN 机群裂纹预后——高级模型，仅在有完整物理参数时可用。"""

    asset_id = "reliability_model.crack.pinn_fleet_prognosis"

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        return {"ok": True, "issues": ["PINN requires TensorFlow + physics parameters + GPU"]}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        return AssetRunResult(
            status="needs_review",
            structured_output={
                "method": "unavailable_experimental",
                "note": "PINN fleet prognosis is L3_OPTIONAL/P4_DEFERRED. "
                        "Requires: TensorFlow, Paris Law physical parameters for specific alloy, "
                        "fleet loading spectra. MIT license. "
                        "git clone https://github.com/PML-UCF/pinn",
            },
            warnings=["Not yet implemented — use FrameworkFDPP or py_fatigue for crack growth"],
            metrics={},
        )


class ChangePointLSTRUL(ImplementationBase):
    """ChangePoint-LSTM 多变工况 RUL——回退到分段线性退化。"""

    asset_id = "reliability_model.rul.changepoint_lstm_multicondition"

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        params = {**self.default_params(), **parameters}
        data = inputs[0] if inputs else {}

        # 从传感器数据中检测退化趋势变化点
        change_points = []
        rul_per_segment = []
        for key, val in data.items():
            if isinstance(val, (list, np.ndarray)) and len(val) > 20:
                arr = np.asarray(val, dtype=np.float64)
                # 简单分段线性检测
                seg_len = len(arr) // 3
                slopes = []
                for s in range(3):
                    seg = arr[s * seg_len:(s + 1) * seg_len]
                    slope = np.polyfit(np.arange(len(seg)), seg, 1)[0] if len(seg) > 2 else 0
                    slopes.append(slope)
                if abs(slopes[-1]) > abs(slopes[0]) * 2:
                    change_points.append({
                        "sensor": key,
                        "segment": 2,
                        "description": f"Degradation rate increased: {slopes[0]:.4f} → {slopes[-1]:.4f}",
                    })

        return AssetRunResult(
            status="success",
            structured_output={
                "change_points_detected": len(change_points),
                "change_points": change_points,
                "method": "segmented_linear_baseline",
                "note": "For ChangePoint-LSTM: git clone https://github.com/en-research/ChangePoint-LSTM",
            },
            metrics={"change_point_count": float(len(change_points))},
        )

    @staticmethod
    def default_params() -> dict:
        return {"changepoint_penalty": 5, "lstm_units": 128, "sequence_length": 50, "min_segment_length": 5}
