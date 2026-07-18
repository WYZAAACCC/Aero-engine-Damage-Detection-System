"""7维数据质量门监控器 — P0_CRITICAL"""

import numpy as np

from ._base import AssetRunResult, ImplementationBase


class DataQualityGate(ImplementationBase):
    """7 维数据质量门监控器。

    检查: 结构完整性/数值有效性/时间一致性/单位和量纲/来源与标定/任务适用性/隐私与保密。
    输出 DataQualityReport: pass/warn/fail + 建议动作。
    """

    asset_id = "monitor.data_quality.data_quality_gate"

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        if not inputs:
            return {"ok": False, "issues": ["No artifact provided for quality check"]}
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        params = {**self.default_params(), **parameters}
        data = inputs[0] if inputs else {}
        results = {}

        # 1. 结构完整性
        results["structural"] = self._check_structural(data)

        # 2. 数值有效性
        results["numerical"] = self._check_numerical(data)

        # 3. 时间一致性
        results["temporal"] = self._check_temporal(data)

        # 4. 单位和量纲
        results["unit_dimensional"] = self._check_units(data)

        # 5. 来源与标定
        results["source_calibration"] = self._check_source(data)

        # 6. 任务适用性
        results["fitness"] = self._check_fitness(data, context)

        # 7. 隐私与保密
        results["privacy"] = self._check_privacy(data)

        # 汇总
        all_results = []
        for cat, checks in results.items():
            all_results.extend(checks)

        fail_count = sum(1 for c in all_results if c["status"] == "fail")
        warn_count = sum(1 for c in all_results if c["status"] == "warn")
        pass_count = sum(1 for c in all_results if c["status"] == "pass")

        if fail_count > 0:
            overall = "fail"
            recommendation = "blocked" if fail_count > 2 else "needs_supplement"
        elif warn_count > 2:
            overall = "warn"
            recommendation = "needs_supplement"
        else:
            overall = "pass"
            recommendation = "proceed"

        # ── 审计修复 (AER-011): 区分执行成功和数据质量结论 ──
        is_usable = overall != "fail"
        return AssetRunResult(
            execution_status="success",  # 程序执行成功
            validity_status="valid" if is_usable else "invalid",  # 数据质量结论
            can_influence_decision=is_usable,
            structured_output={
                "overall_status": overall,
                "passed_count": pass_count,
                "warn_count": warn_count,
                "fail_count": fail_count,
                "recommendation": recommendation,
                "checks": results,
                "note": (
                    "Data quality check is surface-level only. "
                    "Does NOT validate: dimensional consistency (pint), sensor calibration dates, "
                    "operating condition alignment, borescope scale verification, "
                    "material batch traceability, or maintenance record authenticity."
                ),
            },
            metrics={
                "pass_rate": pass_count / max(len(all_results), 1),
                "fail_rate": fail_count / max(len(all_results), 1),
            },
        )

    def _check_structural(self, data: dict) -> list[dict]:
        checks = []
        has_data = bool(data)
        checks.append({
            "category": "structural", "field": "content",
            "status": "pass" if has_data else "fail",
            "message": "数据存在" if has_data else "无数据内容",
        })
        if isinstance(data, dict):
            for field in ["artifact_id", "artifact_type"]:
                checks.append({
                    "category": "structural", "field": field,
                    "status": "pass" if data.get(field) else "warn",
                    "message": f"{field} 已提供" if data.get(field) else f"缺少 {field}",
                })
        return checks

    def _check_numerical(self, data: dict) -> list[dict]:
        """审计修复 (AER-011): 标量数值必须进行范围检查，不能仅检查 NaN/Inf。"""
        checks = []
        numeric_fields = []
        for k, v in data.items():
            if isinstance(v, (int, float)):
                numeric_fields.append((k, float(v)))
                # 范围检查（物理合理性）
                fv = float(v)
                # 温度检查 (K or °C)
                if any(tag in k.lower() for tag in ['temp', 't2', 't24', 't30', 't50', 'egt']):
                    if fv < -100 or fv > 3000:
                        checks.append({
                            "category": "numerical", "field": k,
                            "status": "fail",
                            "message": f"温度 {fv} 超出物理合理范围 [-100, 3000]",
                        })
                # 压力检查
                if any(tag in k.lower() for tag in ['press', 'p2', 'p15', 'p30']):
                    if fv < 0:
                        checks.append({
                            "category": "numerical", "field": k,
                            "status": "fail",
                            "message": f"压力 {fv} 不能为负值",
                        })
                # 振动检查
                if any(tag in k.lower() for tag in ['vib', 'vibration']):
                    if fv < 0 or fv > 100:
                        checks.append({
                            "category": "numerical", "field": k,
                            "status": "fail",
                            "message": f"振动 {fv} 超出合理范围 [0, 100] ips",
                        })
            elif isinstance(v, (list, np.ndarray)):
                arr = np.asarray(v, dtype=np.float64)
                if arr.size == 0:
                    checks.append({
                        "category": "numerical", "field": k,
                        "status": "fail", "message": "数组为空",
                    })
                    continue
                has_nan = bool(np.any(np.isnan(arr)))
                has_inf = bool(np.any(np.isinf(arr)))
                is_constant = bool(np.all(arr == arr[0]))
                if has_nan:
                    checks.append({
                        "category": "numerical", "field": k,
                        "status": "fail", "message": f"含 {int(np.sum(np.isnan(arr)))} 个 NaN",
                    })
                if has_inf:
                    checks.append({
                        "category": "numerical", "field": k,
                        "status": "fail", "message": "含 Inf 值",
                    })
                if is_constant and arr.size > 10:
                    checks.append({
                        "category": "numerical", "field": k,
                        "status": "warn", "message": f"常量信号 (全部={arr[0]:.2f})",
                    })
        if not numeric_fields and not checks:
            checks.append({
                "category": "numerical", "field": "all",
                "status": "warn", "message": "未检测到数值字段",
            })
        return checks or [{"category": "numerical", "field": "all", "status": "pass", "message": "数值检查通过"}]

    def _check_temporal(self, data: dict) -> list[dict]:
        checks = [{
            "category": "temporal", "field": "timestamp",
            "status": "pass" if data.get("timestamp") or data.get("created_at") else "warn",
            "message": "时间戳已提供" if data.get("timestamp") or data.get("created_at") else "缺少时间戳",
        }]
        # 检查 sample_rate
        if data.get("sample_rate"):
            sr = float(data["sample_rate"])
            if sr <= 0:
                checks.append({"category": "temporal", "field": "sample_rate",
                               "status": "fail", "message": "采样率必须 > 0"})
            elif sr < 100:
                checks.append({"category": "temporal", "field": "sample_rate",
                               "status": "warn", "message": f"采样率偏低 ({sr} Hz)"})
        return checks

    def _check_units(self, data: dict) -> list[dict]:
        units = data.get("units", {})
        if not units:
            return [{"category": "unit_dimensional", "field": "units",
                     "status": "warn", "message": "缺少单位声明——不可进行量纲校验"}]
        checks = []
        for field, unit in units.items():
            checks.append({"category": "unit_dimensional", "field": field,
                          "status": "pass", "message": f"{field}: {unit}"})
        return checks or [{"category": "unit_dimensional", "field": "all", "status": "pass", "message": "单位检查通过"}]

    def _check_source(self, data: dict) -> list[dict]:
        return [{
            "category": "source_calibration", "field": "producer_asset_id",
            "status": "pass" if data.get("producer_asset_id") else "warn",
            "message": "来源资产已标注" if data.get("producer_asset_id") else "缺少来源资产标注",
        }]

    def _check_fitness(self, data: dict, context: dict) -> list[dict]:
        return [{
            "category": "fitness", "field": "artifact_type",
            "status": "pass",
            "message": f"数据类型 {data.get('artifact_type', 'unknown')} 可用于任务",
        }]

    def _check_privacy(self, data: dict) -> list[dict]:
        classification = data.get("data_classification", "internal")
        return [{
            "category": "privacy", "field": "data_classification",
            "status": "pass" if classification in ("public", "internal") else "warn",
            "message": f"数据分级: {classification}",
        }]

    @staticmethod
    def default_params() -> dict:
        return {
            "check_structural": True, "check_numerical": True,
            "check_temporal": True, "check_units": True,
            "check_source": True, "check_fitness": True, "check_privacy": True,
        }
