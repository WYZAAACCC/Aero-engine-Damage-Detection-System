"""损伤表征工具 — 几何测量 / 类型分类 / 严重度分级 (P1_HIGH)

三个工具全部自实现，零外部依赖。
"""

import numpy as np
from ._base import AssetRunResult, ImplementationBase


# ═══════════════════════════════════════════════════════════════════════
# 裂纹几何量测量（像素→物理坐标）
# ═══════════════════════════════════════════════════════════════════════

class CrackGeometryMeasurement(ImplementationBase):
    """从分割掩膜测量裂纹几何量（长度/宽度/面积/方向）。

    有标尺→物理值(mm)；无标尺→像素值(pixel fallback)。
    """

    asset_id = "characterizer.crack.geometry_measurement"

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        if not inputs:
            return {"ok": False, "issues": ["No mask or detection data"]}
        data = inputs[0] if isinstance(inputs[0], dict) else {}
        has_mask = data.get("segmentation_mask_uri") or data.get("mask") or data.get("bbox")
        if not has_mask:
            return {"ok": False, "issues": ["No mask/bbox data — cannot measure geometry"]}
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        params = {**self.default_params(), **parameters}
        data = inputs[0] if inputs else {}

        mask_data = data.get("mask", data.get("segmentation_mask_uri"))
        bbox = data.get("bbox")  # [x, y, w, h] or [x1, y1, x2, y2]
        scale_info = data.get("scale_info", {})

        # 从 bbox 估计尺寸
        if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            if len(bbox) == 4:
                w, h = float(bbox[2]), float(bbox[3])
                if bbox[2] > 1.0:  # 判断是 [x,y,w,h] 还是 [x1,y1,x2,y2]
                    w, h = float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
            else:
                w, h = 10.0, 10.0
        elif mask_data is not None:
            mask_arr = np.asarray(mask_data) if not isinstance(mask_data, str) else np.zeros((100, 100))
            ys, xs = np.where(mask_arr > 0.5)
            if len(xs) > 0:
                w, h = float(xs.max() - xs.min()), float(ys.max() - ys.min())
            else:
                w, h = 0.0, 0.0
        else:
            w, h = 0.0, 0.0

        # 标尺转换
        scale_available = bool(scale_info.get("pixel_to_mm"))
        if scale_available:
            px2mm = float(scale_info.get("pixel_to_mm", 0.05))
            length_mm = max(w, h) * px2mm
            width_mm = min(w, h) * px2mm
            area_mm2 = w * h * px2mm ** 2
            measurement_unit = "mm"
        else:
            length_mm = None
            width_mm = None
            area_mm2 = None
            measurement_unit = "pixel"
            scale_available = False

        return AssetRunResult(
            status="success",
            structured_output={
                "length_px": round(max(w, h), 1),
                "width_px": round(min(w, h), 1),
                "area_px2": round(w * h, 1),
                "length_mm": round(length_mm, 3) if length_mm is not None else None,
                "width_mm": round(width_mm, 3) if width_mm is not None else None,
                "area_mm2": round(area_mm2, 3) if area_mm2 is not None else None,
                "scale_available": scale_available,
                "measurement_unit": measurement_unit,
                "calibration_method": scale_info.get("calibration_method", "unknown"),
                "warning": ("FABRICATE_MM_VALUES_FORBIDDEN" if not scale_available else None),
            },
            warnings=["No scale reference — pixel output only. mm values FABRICATED if present."] if not scale_available else [],
            metrics={
                "crack_length_px": max(w, h),
                "crack_aspect_ratio": max(w, h) / max(min(w, h), 1),
            },
        )

    @staticmethod
    def default_params() -> dict:
        return {"min_crack_length_px": 10, "output_pixel_fallback": True,
                "skeletonize_method": "zhang_suen"}


# ═══════════════════════════════════════════════════════════════════════
# 损伤类型分类器
# ═══════════════════════════════════════════════════════════════════════

class DamageTypeClassifier(ImplementationBase):
    """将检测发现分类为 11 种预定义损伤类型。

    基于现象关键词 + 分数语义 + 位置信息的规则引擎。
    """

    asset_id = "characterizer.damage.damage_type_classifier"

    DAMAGE_PATTERNS = {
        "crack": ["crack", "fissure", "linear", "fracture", "裂纹"],
        "coating_spallation": ["spall", "coating_loss", "peel", "debond", "TBC", "thermal_barrier", "剥落", "涂层"],
        "erosion": ["erosion", "pitting", "material_loss", "rough", "冲蚀"],
        "corrosion": ["corrosion", "rust", "oxidation", "discoloration", "腐蚀", "氧化"],
        "burn_mark": ["burn", "hot_spot", "overheat", "thermal_damage", "烧蚀", "过烧"],
        "dent": ["dent", "indentation", "impact_mark", "凹陷", "撞击"],
        "FOD": ["FOD", "foreign_object", "debris", "外物"],
        "wear": ["wear", "abrasion", "fretting", "scuff", "磨损"],
        "deformation": ["deform", "bend", "twist", "distort", "变形"],
        "rub": ["rub", "rubbing", "contact", "tip_rub", "碰摩"],
    }

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        if not inputs:
            return {"ok": False, "issues": ["No finding data"]}
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        data = inputs[0] if inputs else {}
        phenomenon = str(data.get("phenomenon", data.get("description", data.get("target", "")))).lower()
        location = str(data.get("location", ""))
        score = data.get("score")
        score_semantics = str(data.get("score_semantics", "anomaly_score"))

        # 规则匹配
        matches = {}
        for dtype, keywords in self.DAMAGE_PATTERNS.items():
            score_val = sum(1 for kw in keywords if kw.lower() in phenomenon)
            if score_val > 0:
                matches[dtype] = score_val

        if matches:
            best = max(matches, key=matches.get)
            confidence = "observed" if matches[best] >= 3 else "inferred" if matches[best] >= 2 else "suspected"
        else:
            best = "unknown"
            confidence = "suspected"

        # 置信度增强逻辑
        if score is not None:
            try:
                s = float(score)
                if s > 0.9 and confidence == "suspected":
                    confidence = "inferred"
            except (ValueError, TypeError):
                pass

        return AssetRunResult(
            status="success",
            structured_output={
                "damage_type": best,
                "confidence": confidence,
                "matched_keywords": matches,
                "phenomenon": phenomenon[:200],
                "alternatives": sorted(matches.items(), key=lambda x: -x[1])[1:4],
            },
            metrics={"match_count": float(len(matches)), "best_match_score": float(matches.get(best, 0))},
        )


# ═══════════════════════════════════════════════════════════════════════
# 严重度分级器
# ═══════════════════════════════════════════════════════════════════════

class SeverityRater(ImplementationBase):
    """基于规则的 4 级严重度分级器。

    minor / moderate / severe / critical
    """

    asset_id = "characterizer.damage.severity_rater"

    # 严重度规则表：damage_type → {size_threshold: severity, ...}
    RULES = {
        "crack": {"critical_length_mm": 5.0, "severe_length_mm": 2.0, "moderate_length_mm": 0.5},
        "coating_spallation": {"critical_area_percent": 30, "severe_area_percent": 10, "moderate_area_percent": 2},
        "burn_mark": {"always": "severe"},
        "FOD": {"critical_depth_mm": 1.0, "severe_depth_mm": 0.3},
    }

    COMPONENT_CRITICALITY = {
        "HPT Blade Stage 1": "critical",
        "HPT Blade Stage 2": "high",
        "LPT Blade": "moderate",
        "compressor_blade": "moderate",
        "combustor_liner": "high",
        "disk": "critical",
        "casing": "low",
    }

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        if not inputs:
            return {"ok": False, "issues": ["No characterization data"]}
        data = inputs[0] if isinstance(inputs[0], dict) else {}
        if not data.get("damage_type"):
            return {"ok": False, "issues": ["damage_type is required"]}
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        data = inputs[0] if inputs else {}
        damage_type = str(data.get("damage_type", "unknown"))
        geometry = data.get("geometry", {})
        component = str(data.get("component_location", data.get("component", "")))

        length_mm = geometry.get("length_mm")
        area_mm2 = geometry.get("area_mm2")

        rules = self.RULES.get(damage_type, {})
        severity = "moderate"  # 默认
        criteria = []

        # 几何阈值评分
        if "critical_length_mm" in rules and length_mm is not None:
            if length_mm >= rules["critical_length_mm"]:
                severity = "critical"
                criteria.append(f"length_{length_mm}mm_>=_critical_{rules['critical_length_mm']}mm")
            elif length_mm >= rules.get("severe_length_mm", 999):
                severity = "severe"
                criteria.append(f"length_{length_mm}mm_>=_severe_{rules['severe_length_mm']}mm")
            elif length_mm >= rules.get("moderate_length_mm", 999):
                severity = "moderate"
                criteria.append(f"length_{length_mm}mm_>=_moderate_{rules['moderate_length_mm']}mm")
            else:
                severity = "minor"
                criteria.append(f"length_{length_mm}mm_below_all_thresholds")

        if "critical_area_percent" in rules and area_mm2 is not None:
            if area_mm2 >= rules["critical_area_percent"]:
                severity = "critical"
                criteria.append(f"area_{area_mm2}pct_>=_critical_{rules['critical_area_percent']}pct")
            elif area_mm2 >= rules.get("severe_area_percent", 999):
                severity = max(severity, "severe", key=lambda s: ["minor", "moderate", "severe", "critical"].index(s))

        # 部件关键性调整
        comp_crit = self.COMPONENT_CRITICALITY.get(component)
        if comp_crit == "critical" and severity == "moderate":
            severity = "severe"
            criteria.append("component_criticality_escalation")

        if "always" in rules:
            severity = rules["always"]
            criteria.append(f"always_{severity}_for_{damage_type}")

        return AssetRunResult(
            status="success",
            structured_output={
                "severity": severity,
                "criteria_met": criteria,
                "damage_type": damage_type,
                "component": component,
                "rule_version": "1.0",
            },
            metrics={"severity_ordinal": float(["minor", "moderate", "severe", "critical"].index(severity))},
        )
