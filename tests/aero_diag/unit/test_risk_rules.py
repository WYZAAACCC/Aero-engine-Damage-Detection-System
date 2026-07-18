"""风险分级与知识检索单元测试"""
import sys
sys.path.insert(0, 'src')

from aero_diag.plugins.official.implementations.knowledge_and_rules import (
    RiskClassificationRules, InspectionIntervalRules, OminKnowledgeSource,
)


class TestRiskClassification:
    """风险分级测试 (AER-012)"""

    def test_known_severity_returns_valid(self):
        """已知severity+组件应返回有效风险"""
        rules = RiskClassificationRules()
        result = rules.run(
            inputs=[{'severity': 'critical', 'damage_type': 'crack',
                     'component': 'HPT_blade'}],
            parameters={}, context={},
        )
        assert result.structured_output["risk_level"] in (
            "critical", "high", "medium", "low", "negligible"
        )

    def test_unknown_severity_returns_blocked(self):
        """未知severity应返回unknown+blocked"""
        rules = RiskClassificationRules()
        result = rules.run(
            inputs=[{'severity': 'extreme', 'damage_type': 'crack',
                     'component': 'HPT_blade'}],
            parameters={}, context={},
        )
        assert result.structured_output["risk_level"] == "unknown"
        assert result.structured_output["decision_blocked"] is True
        assert result.structured_output["requires_review"] is True

    def test_unknown_component_returns_blocked(self):
        """完全不匹配的组件应返回unknown"""
        rules = RiskClassificationRules()
        result = rules.run(
            inputs=[{'severity': 'severe', 'damage_type': 'crack',
                     'component': 'non_existent_component'}],
            parameters={}, context={},
        )
        assert result.structured_output["risk_level"] == "unknown"
        assert result.structured_output["decision_blocked"] is True

    def test_all_results_require_review(self):
        """审计修复: 所有结果应标记 requires_review=True"""
        rules = RiskClassificationRules()
        for sev in ["minor", "moderate", "severe", "critical"]:
            result = rules.run(
                inputs=[{'severity': sev, 'damage_type': 'crack',
                         'component': 'HPT_blade'}],
                parameters={}, context={},
            )
            if result.structured_output["risk_level"] != "unknown":
                assert result.structured_output["requires_review"] is True, \
                    f"severity={sev} should require review"


class TestKnowledgeFiltering:
    """知识检索测试 (AER-013)"""

    def test_engine_filter_reduces_results(self):
        """发动机型号硬过滤应减少结果"""
        kb = OminKnowledgeSource()
        r1 = kb.run(
            inputs=[{'query': 'HPT blade crack'}],
            parameters={'component': 'HPT_blade', 'damage': 'crack',
                        'engine_model': 'CFM56-7B', 'material': 'nickel_superalloy'},
            context={},
        )
        r2 = kb.run(
            inputs=[{'query': 'HPT blade crack'}],
            parameters={'component': 'HPT_blade', 'damage': 'crack',
                        'engine_model': 'unrelated_engine', 'material': 'wood'},
            context={},
        )
        assert r2.structured_output["results_found"] <= r1.structured_output["results_found"], \
            "Hard filter should not return MORE results for wrong engine/material"

    def test_cross_domain_tbc_on_fan(self):
        """TBC在风扇叶片上的跨域检测应触警"""
        kb = OminKnowledgeSource()
        result = kb.run(
            inputs=[{'query': 'TBC coating spallation crack'}],
            parameters={'component': 'fan_blade'},
            context={},
        )
        warnings = result.structured_output.get("cross_domain_warnings", [])
        assert len(warnings) > 0, "TBC on fan blade should trigger cross-domain warning"


class TestInspectionInterval:
    """复检周期测试"""

    def test_interval_marked_degraded(self):
        """复检周期应标记为degraded"""
        rules = InspectionIntervalRules()
        result = rules.run(
            inputs=[{'risk_level': 'high', 'damage_type': 'crack'}],
            parameters={}, context={},
        )
        assert result.validity_status == "degraded"
        assert result.can_influence_decision is False
