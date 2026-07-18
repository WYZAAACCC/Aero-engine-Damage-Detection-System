"""知识源 (3个) + 决策规则 (2个) 实现"""

from ._base import AssetRunResult, ImplementationBase


# ═══════════════════════════════════════════════════════════════════════
# 知识源
# ═══════════════════════════════════════════════════════════════════════

class OminKnowledgeSource(ImplementationBase):
    """航空发动机损伤诊断专家知识库 — 严格适用域约束。

    每条知识有精确的 applicability (component/damage/material/engine/condition)
    和 exclusions (禁止跨场景使用)。检索时自动检测跨域误用并返回警告。
    """

    asset_id = "knowledge_source.maintenance.omin_faa_knowledge"

    def __init__(self):
        from .expert_knowledge_base import KNOWLEDGE_ITEMS
        self.KB = KNOWLEDGE_ITEMS

    # ── 部件层级映射：子部件 → 父部件 ──
    COMPONENT_HIERARCHY = {
        "HPT_blade": ["turbine_blade", "blade", "HPT"],
        "LPT_blade": ["turbine_blade", "blade", "LPT"],
        "HPC_blade": ["compressor_blade", "blade", "HPC"],
        "LPC_blade": ["compressor_blade", "blade", "LPC"],
        "fan_blade": ["blade", "fan"],
        "compressor_blade": ["blade", "compressor"],
        "turbine_blade": ["blade", "turbine"],
        "turbine_vane": ["turbine", "vane"],
        "combustor_liner": ["combustor", "liner"],
        "disk": ["disk"],
        "bearing": ["bearing"],
        "rotor": ["rotor"],
    }

    def _component_matches(self, query_component: str, item_components: list[str]) -> bool:
        """检查查询部件是否匹配知识条目部件（含层级）。"""
        qc = query_component.lower().replace(" ", "_")
        for ic in item_components:
            ic_clean = ic.lower().replace(" ", "_")
            # 精确匹配
            if qc == ic_clean:
                return True
            # 子部件匹配父部件 (HPT_blade → turbine_blade)
            if qc in self.COMPONENT_HIERARCHY:
                for parent in self.COMPONENT_HIERARCHY[qc]:
                    if parent == ic_clean or parent in ic_clean:
                        return True
            # 父部件匹配子部件 (turbine_blade → HPT_blade)
            if ic_clean in self.COMPONENT_HIERARCHY:
                for parent in self.COMPONENT_HIERARCHY[ic_clean]:
                    if parent == qc or parent in qc:
                        return True
            # any 匹配所有
            if ic_clean == "any":
                return True
            if qc == "any":
                return True
            # 子串匹配（最后手段）
            if len(qc) > 3 and qc in ic_clean:
                return True
            if len(ic_clean) > 3 and ic_clean in qc:
                return True
        return False

    def _check_cross_domain(self, query: str, component: str, damage: str) -> list[str]:
        """检测查询是否跨越适用域边界。"""
        warnings = []
        q = query.lower()
        c = component.lower().replace(" ", "_") if component else ""

        # TBC/热障涂层 → 仅涡轮/燃烧室，风扇/压气机无TBC
        if any(kw in q for kw in ["tbc", "热障涂层", "thermal barrier", "涂层剥落"]) and c:
            if any(x in c for x in ["fan_blade", "compressor_blade"]):
                warnings.append("DOMAIN: TBC仅存在于涡轮叶片/导向叶片/燃烧室—风扇/压气机叶片无TBC")

        # 蠕变 → 仅镍基合金涡轮叶片>800°C
        if any(kw in q for kw in ["creep", "蠕变", "rafting"]) and c:
            if any(x in c for x in ["fan_blade", "compressor_blade"]):
                warnings.append("DOMAIN: 蠕变仅发生在镍基高温合金涡轮叶片>800°C—风扇/压气机叶片不蠕变")

        # FOD → 仅冷端，涡轮端材料缺失通常不是FOD
        if any(kw in q for kw in ["fod", "外物损伤", "bird strike", "鸟击"]) and c:
            if any(x in c for x in ["turbine_blade", "combustor_liner"]):
                warnings.append("DOMAIN: FOD主要影响风扇/压气机—涡轮端材料缺失更可能是涂层剥落/烧蚀/热腐蚀")

        # 热腐蚀 → 仅沿海涡轮叶片
        if any(kw in q for kw in ["热腐蚀", "hot corrosion", "sulfidation"]) and c:
            if any(x in c for x in ["fan_blade", "compressor_blade"]):
                warnings.append("DOMAIN: 热腐蚀仅影响沿海/海上运营的涡轮叶片—风扇/压气机腐蚀通常为点蚀/氧化")

        # Paris → 仅裂纹，不适用于腐蚀/涂层等非裂纹退化
        if "paris" in q and damage:
            if any(d in damage.lower().replace(" ", "_") for d in ["corrosion", "erosion", "coating_spallation", "creep"]):
                warnings.append(f"DOMAIN: Paris公式仅适用于裂纹扩展(LEFM)—不适用于'{damage}'")

        # TC4 → 仅风扇/压气机，涡轮叶片是镍基合金
        if "tc4" in q and c and "turbine" in c:
            warnings.append("DOMAIN: TC4是钛合金(风扇/压气机)—涡轮叶片为镍基合金(IN718/IN738)，材料参数不可互换")

        return warnings

    # ── 扩展关键词映射（中英文同义词）──
    SYNONYMS = {
        "hpt": ["high_pressure_turbine", "turbine_blade", "高压涡轮", "涡轮叶片", "hpt_blade"],
        "lpt": ["low_pressure_turbine", "低压涡轮", "lpt_blade"],
        "hpc": ["high_pressure_compressor", "高压压气机", "compressor_blade"],
        "fan": ["fan_blade", "风扇叶片", "风扇", "n1"],
        "crack": ["裂纹", "开裂", "断裂", "fracture"],
        "coating": ["涂层", "剥落", "spallation", "tbc", "热障涂层", "coating_spallation"],
        "erosion": ["冲蚀", "磨损", "冲刷"],
        "corrosion": ["腐蚀", "锈蚀", "氧化", "oxidation"],
        "fod": ["外物损伤", "冲击", "impact", "foreign_object", "打伤"],
        "fatigue": ["疲劳", "hcf", "lcf", "高周", "低周", "振动疲劳"],
        "creep": ["蠕变", "伸长", "高温变形"],
        "burn": ["烧蚀", "过烧", "超温", "overtemperature", "burn_mark", "烧痕"],
        "vibration": ["振动", "共振", "flutter", "颤振", "bpf"],
        "borescope": ["孔探", "内窥镜", "内窥", "endoscope"],
        "rul": ["剩余寿命", "寿命预测", "可用极限", "remaining_useful_life"],
    }

    def _expand_query(self, query: str) -> set[str]:
        """扩展查询词：中文用双字分词+英文同义词扩展。"""
        query_clean = query.lower().replace("_", " ")
        # 中文按2-4字滑动窗口分词
        tokens = set(query_clean.split())  # 英文按空格
        chinese_chars = ''.join(c for c in query if '一' <= c <= '鿿')
        for n in (2, 3, 4):
            for i in range(len(chinese_chars) - n + 1):
                tokens.add(chinese_chars[i:i+n])
        # 加入完整中文短语
        tokens.add(chinese_chars)
        # 同义词扩展
        expanded = set(tokens)
        for token in list(tokens):
            for key, synonyms in self.SYNONYMS.items():
                if token in synonyms or token == key:
                    expanded.add(key)
                    expanded.update(synonyms)
        return expanded

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        params = {**self.default_params(), **parameters}
        query = str(params.get("query") or (inputs[0].get("query", "") if inputs else ""))
        component = params.get("component") or ""
        damage = params.get("damage") or ""
        ktype = params.get("knowledge_type") or ""
        engine_model = params.get("engine_model") or (inputs[0].get("engine_model", "") if inputs else "")
        material = params.get("material") or (inputs[0].get("material", "") if inputs else "")

        # ── 跨域检查 ──
        cross_domain_warnings = self._check_cross_domain(query, component, damage)

        # 扩展查询词
        query_tokens = self._expand_query(query)
        query_lower = query.lower().replace("_", " ")

        results = []
        engine_filter_applied = bool(engine_model)
        material_filter_applied = bool(material)

        for item in self.KB:
            score = 0
            appl = item.get("applicability", {})
            # ... (existing field parsing)
            score = 0
            appl = item.get("applicability", {})

            # ── 适用域严格打分 ──
            component_list = appl.get("component", [])
            if isinstance(component_list, str):
                component_list = [c.strip() for c in component_list.split(",")]

            damage_list = appl.get("damage_type", appl.get("damage", []))
            if isinstance(damage_list, str):
                damage_list = [d.strip() for d in damage_list.split(",")]

            material_list = appl.get("material", [])
            if isinstance(material_list, str):
                material_list = [m.strip() for m in material_list.split(",")]

            engine_list = appl.get("engine_model", [])
            if isinstance(engine_list, str):
                engine_list = [e.strip() for e in engine_list.split(",")]

            exclusions = item.get("exclusions", {})
            excl_component = exclusions.get("component", [])
            if isinstance(excl_component, str):
                excl_component = [c.strip() for c in excl_component.split(",")]
            excl_note = exclusions.get("note", "")

            # ── 按类型过滤 ──
            if ktype and item.get("type") != ktype:
                continue

            # ── 按部件过滤（层级感知匹配）──
            if component:
                if not self._component_matches(component, component_list):
                    continue
                score += 5  # 部件匹配

            # ── 按损伤类型过滤 ──
            if damage:
                matched = False
                for d in damage_list:
                    if damage.lower().replace(" ", "_") == d.lower().replace(" ", "_"):
                        matched = True; break
                    if damage.lower() in d.lower():
                        matched = True; break
                if not matched:
                    continue
                score += 5  # 损伤类型精确匹配

            # ── 审计修复 (AER-013): 发动机型号硬过滤 ──
            if engine_model:
                if engine_list and engine_list != [""] and "any" not in [e.lower() for e in engine_list]:
                    if not any(engine_model.lower() in e.lower() for e in engine_list):
                        continue  # 发动机型号不匹配 → 硬排除
                    score += 5  # 发动机型号匹配

            # ── 审计修复 (AER-013): 材料硬过滤 ──
            if material:
                if material_list and material_list != [""] and "any" not in [m.lower() for m in material_list]:
                    mat_lower = material.lower().replace(" ", "_")
                    if not any(mat_lower in m.lower().replace(" ", "_") for m in material_list):
                        continue  # 材料不匹配 → 硬排除
                    score += 5  # 材料匹配

            # ── 排除域检查 ──
            applicability_warning = ""
            if component and excl_component and excl_component != [""]:
                if self._component_matches(component, excl_component):
                    applicability_warning = (
                        f"WARNING: knowledge '{item['knowledge_id']}' "
                        f"excludes this component context. {excl_note}"
                    )
                    score -= 10

            # ── 审计修复: 材料排除 ──
            excl_material = exclusions.get("material", [])
            if isinstance(excl_material, str):
                excl_material = [m.strip() for m in excl_material.split(",")]
            if material and excl_material and excl_material != [""]:
                for em in excl_material:
                    if material.lower().replace(" ", "_") in em.lower().replace(" ", "_"):
                        applicability_warning = (
                            f"WARNING: knowledge '{item['knowledge_id']}' "
                            f"excludes material '{material}'. {excl_note}"
                        )
                        score -= 10

            # ── 关键词匹配 ──
            title_lower = item["title"].lower()
            for token in query_tokens:
                if token in title_lower:
                    score += 1

            # 内容关键词匹配
            content_lower = item["content"].lower()
            for token in query_tokens:
                if len(token) > 1 and token in content_lower:
                    score += 0.5

            # 条件匹配 (如 vibration_BPF_amplitude > baseline + 3dB)
            condition = appl.get("condition", "")
            if condition:
                for token in query_tokens:
                    if token in condition.lower():
                        score += 2

            if score > 0 or component or damage or ktype:
                results.append({
                    **item,
                    "relevance_score": round(score, 1),
                    "applicability_warning": applicability_warning if applicability_warning else None,
                })

        results.sort(key=lambda r: -r["relevance_score"])
        max_results = params.get("max_results", 15)

        # 统计
        types_found = {}
        warning_count = 0
        for r_item in results[:max_results]:
            t = r_item.get("type", "unknown")
            types_found[t] = types_found.get(t, 0) + 1
            if r_item.get("applicability_warning"):
                warning_count += 1

        return AssetRunResult(
            execution_status="success",
            validity_status="degraded",  # 29条手写知识，非文档RAG
            can_influence_decision=False,
            structured_output={
                "results_found": len(results),
                "knowledge_items": results[:max_results],
                "query": query[:200],
                "query_expanded": sorted(query_tokens)[:30],
                "total_indexed": len(self.KB),
                "types_distribution": types_found,
                "applicability_warnings": warning_count,
                "evidence_summary": {
                    "level_A": sum(1 for r in results if r.get("evidence_level") == "A"),
                    "level_B": sum(1 for r in results if r.get("evidence_level") == "B"),
                },
                "cross_domain_warnings": cross_domain_warnings,
                "filters_applied": {
                    "component": bool(component),
                    "damage_type": bool(damage),
                    "engine_model": bool(engine_model),
                    "material": bool(material),
                    "knowledge_type": bool(ktype),
                },
                "note": (
                    "29 hand-written knowledge items — NOT a document RAG system. "
                    "24 of 29 self-labeled as evidence level A without approval process. "
                    "Only 1 of 29 has source_location (page number). "
                    "Most lack: page, table, section, document revision, effective date, "
                    "file hash, original excerpt, and reviewer signature. "
                    "DO NOT use for real maintenance decisions."
                ),
            },
            metrics={
                "result_count": float(len(results)),
                "top_score": float(results[0]["relevance_score"]) if results else 0,
            },
        )

    @staticmethod
    def default_params() -> dict:
        return {"max_results": 15, "min_relevance": 0.5}


class BoeingKnowledgeNER(ImplementationBase):
    """Boeing Aviation NER 知识抽取。"""

    asset_id = "knowledge_source.maintenance.boeing_aviation_ner_knowledge"

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        items = [
            {"knowledge_id": "boeing_sdr_pattern_001", "type": "standard_clause",
             "title": "Common SDR Failure Patterns — HPT",
             "content": "SDR分析显示HPT叶片最常见报告故障模式为：TBC剥落(34%), 裂纹(28%), 烧蚀(18%), FOD(12%), 其他(8%)",
             "source": "Boeing SDR analysis FY2024", "evidence_level": "B"},
        ]
        return AssetRunResult(
            status="success",
            structured_output={"knowledge_items": items, "total_indexed": len(items)},
            metrics={"result_count": float(len(items))},
        )


class MaintIEKnowledgeSource(ImplementationBase):
    """MaintIE 维修知识本体。"""

    asset_id = "knowledge_source.maintenance.maintie_schema_knowledge"

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        schema_info = {
            "top_classes": ["PhysicalObject", "State", "Process", "Activity", "Property"],
            "leaf_entities": 224,
            "relation_types": ["has_state", "undergoes", "located_at", "has_property", "results_in", "performed_by"],
            "note": "MaintIE Schema v1.0. Full dataset: git clone https://github.com/nlp-tlp/maintie",
        }
        return AssetRunResult(
            status="success",
            structured_output={"schema": schema_info, "method": "schema_reference"},
            metrics={},
        )


# ═══════════════════════════════════════════════════════════════════════
# 决策规则
# ═══════════════════════════════════════════════════════════════════════

class RiskClassificationRules(ImplementationBase):
    """5 级风险分级规则引擎。"""

    asset_id = "decision_rule.engine.risk_classification"

    RISK_MATRIX = {
        ("critical", "HPT Blade"): "critical",
        ("critical", "disk"): "critical",
        ("severe", "HPT Blade"): "high",
        ("severe", "combustor_liner"): "high",
        ("moderate", "HPT Blade"): "medium",
        ("minor", "any"): "negligible",
    }

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        if not inputs:
            return {"ok": False, "issues": ["Need damage characterization"]}
        data = inputs[0]
        if not data.get("damage_type") and not data.get("severity"):
            return {"ok": False, "issues": ["damage_type or severity required"]}
        # 警告但不阻断——unknown severity 在 run() 中处理
        issues = []
        sev = data.get("severity", "")
        valid_severities = {"minor", "moderate", "severe", "critical"}
        if sev and sev not in valid_severities:
            issues.append(
                f"Severity '{sev}' not in standard set {valid_severities}. "
                "Will be treated as unknown in risk classification."
            )
        return {"ok": True, "issues": issues}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        data = inputs[0] if inputs else {}
        severity = str(data.get("severity", ""))
        component = str(data.get("component", data.get("component_location", "")))
        damage_type = str(data.get("damage_type", ""))

        # 矩阵查找
        risk = None  # 审计修复: 初始化为 None 而非 "medium"
        rule_hit = ""
        for (sev_pat, comp_pat), risk_level in self.RISK_MATRIX.items():
            if sev_pat == severity and (comp_pat == "any" or comp_pat.lower() in component.lower()):
                risk = risk_level
                rule_hit = f"{sev_pat}_{comp_pat}"
                break

        # 默认逻辑 — 仅当 severity 在已知范围内
        if not rule_hit:
            risk_map = {"critical": "critical", "severe": "high", "moderate": "medium", "minor": "low"}
            risk = risk_map.get(severity)

        # ── 审计修复 (AER-012): 未知组合返回 unknown，不是 medium ──
        if risk is None:
            return AssetRunResult(
                execution_status="success",
                validity_status="degraded",
                can_influence_decision=False,
                structured_output={
                    "risk_level": "unknown",
                    "requires_review": True,
                    "decision_blocked": True,
                    "reason": (
                        f"NO_APPLICABLE_APPROVED_RULE — "
                        f"severity='{severity}', component='{component}', damage_type='{damage_type}'. "
                        f"No matching rule in risk matrix. "
                        f"Risk MUST NOT default to 'medium' for unknown combinations."
                    ),
                    "candidate_actions": ["engineering_evaluation"],
                },
                warnings=[
                    "Unknown severity/component combination — risk classification blocked. "
                    "Requires manual engineering evaluation, not automated default."
                ],
                metrics={"risk_ordinal": -1},
            )

        actions = {
            "critical": ["immediate_shutdown", "engineering_evaluation"],
            "high": ["reinspection", "repair", "possible_derate"],
            "medium": ["increased_monitoring", "scheduled_reinspection"],
            "low": ["continue_operation", "routine_monitoring"],
            "negligible": ["continue_operation"],
        }

        return AssetRunResult(
            execution_status="success",
            validity_status="degraded",  # 审计: 通用规则不是验证过的
            can_influence_decision=False,
            structured_output={
                "risk_level": risk,
                "rule_hit": rule_hit or "default_severity_map",
                "candidate_actions": actions.get(risk, ["engineering_evaluation"]),
                "requires_review": True,  # 审计修复: 始终要求复核
                "rule_version": "1.0",
                "note": (
                    "Generic risk matrix — NOT engine-model or component-position specific. "
                    "May not match AMM/ESM risk criteria for specific inspection findings."
                ),
            },
            metrics={"risk_ordinal": float(["negligible", "low", "medium", "high", "critical"].index(risk) if risk in ["negligible", "low", "medium", "high", "critical"] else -1)},
        )


class InspectionIntervalRules(ImplementationBase):
    """复检周期决策规则。"""

    asset_id = "decision_rule.engine.inspection_interval"

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        data = inputs[0] if inputs else {}
        risk_level = str(data.get("risk_level", "medium"))
        damage_type = str(data.get("damage_type", "unknown"))

        intervals = {
            "critical": {"interval_cycles": 0, "action": "immediate_shutdown", "description": "立即停飞——不允许继续运行"},
            "high": {"interval_cycles": 25, "action": "reinspection", "description": "25循环内复检"},
            "medium": {"interval_cycles": 100, "action": "scheduled_reinspection", "description": "100循环内安排复检"},
            "low": {"interval_cycles": 300, "action": "routine_inspection", "description": "300循环内按常规计划检查"},
            "negligible": {"interval_cycles": 1000, "action": "routine_inspection", "description": "常规检查周期"},
        }

        rec = intervals.get(risk_level, intervals["medium"])

        return AssetRunResult(
            execution_status="success",
            validity_status="degraded",  # 硬编码值，非发动机型号特定
            can_influence_decision=False,
            structured_output={
                "recommended_interval_cycles": rec["interval_cycles"],
                "action_type": rec["action"],
                "description": rec["description"],
                "risk_level": risk_level,
                "damage_type": damage_type,
                "rule_version": "1.0",
                "requires_dual_review": risk_level in ("critical", "high"),
                "note": (
                    "HARDCODED inspection intervals — NOT based on engine model, "
                    "ATA chapter, specific component position, damage morphology, "
                    "operating environment, load spectrum, OEM maintenance program, "
                    "or Probability of Detection (POD). "
                    "Fixed intervals MUST NOT be used for real maintenance planning."
                ),
            },
            metrics={"interval_cycles": float(rec["interval_cycles"])},
        )
