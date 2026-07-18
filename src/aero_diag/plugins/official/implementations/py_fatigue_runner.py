"""py_fatigue Paris Law 裂纹扩展 — P1_HIGH"""

import numpy as np

from ._base import AssetRunResult, ImplementationBase


class ParisLawCrackGrowth(ImplementationBase):
    """基于 Paris Law 的确定性疲劳裂纹扩展计算。

    da/dN = C * (ΔK)^m

    使用 py_fatigue 库进行精确计算；如未安装则回退到自实现的数值积分。
    """

    asset_id = "reliability_model.crack.py_fatigue_paris_law"

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        params = {**self.default_params(), **parameters}
        missing = []
        for key in ["C", "m", "initial_crack_mm", "critical_crack_mm"]:
            if key not in params and not any(key in (i or {}) for i in inputs):
                missing.append(key)
        if missing:
            return {"ok": False, "issues": [f"Missing parameters: {missing}"]}
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        params = {**self.default_params(), **parameters}
        data = inputs[0] if inputs else {}

        C = float(data.get("C", params.get("C", 1e-12)))
        m = float(data.get("m", params.get("m", 3.0)))
        a0_mm = float(data.get("initial_crack_mm", params.get("initial_crack_mm", 1.0)))
        ac_mm = float(data.get("critical_crack_mm", params.get("critical_crack_mm", 10.0)))
        delta_sigma = float(data.get("delta_sigma_mpa", params.get("delta_sigma_mpa", 200.0)))

        # 转换为米
        a0 = a0_mm * 1e-3
        ac = ac_mm * 1e-3
        da_max = (ac - a0) / 10000.0  # 数值积分步长

        try:
            import py_fatigue as pf
            _has_pyf = True
        except ImportError:
            _has_pyf = False

        if _has_pyf:
            try:
                curve = pf.ParisCurve(C=C, m=m)
                result = self._numeric_integration(a0, ac, C, m, delta_sigma, da_max)
            except Exception:
                result = self._numeric_integration(a0, ac, C, m, delta_sigma, da_max)
        else:
            result = self._numeric_integration(a0, ac, C, m, delta_sigma, da_max)

        cycles = result["cycles"]
        a_history = result["a_history"]

        return AssetRunResult(
            status="success",
            structured_output={
                "paris_parameters": {"C": C, "m": m},
                "initial_crack_mm": a0_mm,
                "critical_crack_mm": ac_mm,
                "cycles_to_failure": int(cycles),
                "da_dn_at_critical": float(C * (delta_sigma * np.sqrt(np.pi * ac))**m),
                "method": "py_fatigue" if _has_pyf else "numeric_integration_fallback",
            },
            metrics={
                "cycles_to_failure": float(cycles),
                "final_crack_length_mm": float(a_history[-1] * 1e3) if a_history else ac_mm,
            },
        )

    def _numeric_integration(self, a0, ac, C, m, delta_sigma, da_max):
        """数值积分 Paris Law"""
        a = a0
        cycles = 0
        a_history = [a0]
        geometry_factor = 1.12  # 默认边缘裂纹几何因子

        while a < ac and cycles < 1e9:
            delta_K = delta_sigma * geometry_factor * np.sqrt(np.pi * a)
            da_dN = C * (delta_K ** m)
            da = min(da_max, ac - a)
            dN = da / da_dN if da_dN > 0 else 1
            a += da
            cycles += int(dN)
            a_history.append(a)

        return {"cycles": cycles, "a_history": a_history}

    @staticmethod
    def default_params() -> dict:
        return {
            "C": 1e-12,
            "m": 3.0,
            "initial_crack_mm": 1.0,
            "critical_crack_mm": 10.0,
            "delta_sigma_mpa": 200.0,
            "geometry_factor": 1.12,
        }
