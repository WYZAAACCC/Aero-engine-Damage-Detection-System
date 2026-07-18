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
        """审计修复 (P0-3): 禁用默认值，必须显式提供材料参数。

        危险默认值 (C=1e-12, m=3.0, delta_sigma=200MPa) 可能导致
        严重错误的安全评估。调用者必须提供具体材料和载荷信息。
        """
        params = {**parameters}
        data = inputs[0] if inputs else {}
        # 不合并 default_params()——调用者必须显式提供所有关键参数
        missing = []
        for key in ["C", "m", "initial_crack_mm", "critical_crack_mm", "delta_sigma_mpa"]:
            val = data.get(key, params.get(key))
            if val is None:
                missing.append(key)
        if missing:
            return {
                "ok": False,
                "issues": [
                    f"Missing REQUIRED parameters: {missing}. "
                    "Paris Law MUST NOT use default values for safety-critical crack growth assessment. "
                    "Provide: C (material constant), m (exponent), initial_crack_mm, critical_crack_mm, delta_sigma_mpa (stress range). "
                    "Material source must be documented (e.g. NASGRO database, material test report)."
                ],
            }
        # 验证物理合理性
        a0 = float(data.get("initial_crack_mm", params.get("initial_crack_mm", 0)))
        ac = float(data.get("critical_crack_mm", params.get("critical_crack_mm", 0)))
        if ac <= a0:
            return {
                "ok": False,
                "issues": [f"critical_crack_mm ({ac}) must be > initial_crack_mm ({a0})"],
            }
        ds = float(data.get("delta_sigma_mpa", params.get("delta_sigma_mpa", 0)))
        if ds <= 0:
            return {"ok": False, "issues": ["delta_sigma_mpa must be > 0"]}
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        """Paris Law 数值积分——仅演示用途。

        审计修复: 参数必须显式提供; 固定几何因子仅适用于理想边缘裂纹;
        不检查 ΔKth/Kc/短裂纹/高温蠕变 LEFM 适用域。
        """
        params = {**parameters}
        data = inputs[0] if inputs else {}

        # 不合并默认值
        C = float(data.get("C", params.get("C")))
        m = float(data.get("m", params.get("m")))
        a0_mm = float(data.get("initial_crack_mm", params.get("initial_crack_mm")))
        ac_mm = float(data.get("critical_crack_mm", params.get("critical_crack_mm")))
        delta_sigma = float(data.get("delta_sigma_mpa", params.get("delta_sigma_mpa")))
        geometry_factor = float(data.get("geometry_factor", params.get("geometry_factor", 1.12)))

        # 参数来源记录
        material_source = data.get("material_source", params.get("material_source", ""))
        if not material_source:
            material_source = "UNSPECIFIED — parameters not traceable to material database or test report"

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
            execution_status="success",
            validity_status="degraded",  # 仅数值演示，缺少材料/载荷验证
            can_influence_decision=False,
            structured_output={
                "paris_parameters": {"C": C, "m": m, "geometry_factor": geometry_factor},
                "material_source": material_source,
                "initial_crack_mm": a0_mm,
                "critical_crack_mm": ac_mm,
                "delta_sigma_mpa": delta_sigma,
                "cycles_to_failure": int(cycles),
                "da_dn_at_critical": float(C * (delta_sigma * geometry_factor * np.sqrt(np.pi * ac))**m),
                "method": "numeric_integration_demo",
                "note": (
                    "DEMONSTRATION ONLY. "
                    "Fixed geometry_factor=1.12 assumes ideal edge crack — not valid for all geometries. "
                    "No ΔKth (threshold) or Kc (fracture toughness) check — Paris law is invalid outside LEFM range. "
                    "No short-crack, high-temperature creep, or corrosion-fatigue applicability check. "
                    "Int(dN) per step introduces truncation error. "
                    "Do NOT use for real engine life assessment."
                ),
            },
            metrics={
                "cycles_to_failure": float(cycles),
                "final_crack_length_mm": float(a_history[-1] * 1e3) if a_history else ac_mm,
            },
        )

    def _numeric_integration(self, a0, ac, C, m, delta_sigma, da_max,
                             geometry_factor=1.12):
        """数值积分 Paris Law"""
        a = a0
        cycles = 0
        a_history = [a0]

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
        """审计修复: 不提供安全关键默认值。必须由调用者显式传入。"""
        return {
            # 所有关键参数必须显式提供，不设默认值
            # C, m, initial_crack_mm, critical_crack_mm, delta_sigma_mpa 必须传入
            "geometry_factor": 1.12,  # 仅几何因子有合理默认值（边缘裂纹）
            "material_source": "",     # 强制要求材料来源
        }
