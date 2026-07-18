"""几何测量与表征单元测试 — bbox格式、负值拒绝、量纲检查"""
import pytest
import sys
sys.path.insert(0, 'src')

from aero_diag.plugins.official.implementations.characterizers import (
    CrackGeometryMeasurement, DamageTypeClassifier, SeverityRater,
)


class TestCrackGeometry:
    """bbox格式与不变量测试 (AER-010)"""

    def test_xyxy_format_valid(self):
        """合法 xyxy bbox 应返回有效结果"""
        geom = CrackGeometryMeasurement()
        result = geom.run(
            inputs=[{'bbox_format': 'xyxy', 'bbox': [100, 50, 200, 80],
                     'scale_info': {'pixel_to_mm': 0.05}}],
            parameters={}, context={},
        )
        assert result.execution_status == "success"
        assert result.validity_status == "valid"
        assert result.structured_output["length_px"] > 0
        assert result.structured_output["width_px"] > 0
        assert result.structured_output["area_px2"] > 0

    def test_unknown_format_rejected(self):
        """未知 bbox_format 应被拒绝"""
        geom = CrackGeometryMeasurement()
        result = geom.run(
            inputs=[{'bbox_format': 'unknown', 'bbox': [100, 50, 200, 80]}],
            parameters={}, context={},
        )
        assert result.execution_status == "failed"
        assert result.validity_status == "invalid"

    def test_negative_dimensions_rejected(self):
        """负宽度/负高度应被拒绝"""
        geom = CrackGeometryMeasurement()
        result = geom.run(
            inputs=[{'bbox_format': 'xyxy', 'bbox': [200, 80, 100, 50]}],  # x2<x1, y2<y1
            parameters={}, context={},
        )
        assert result.execution_status == "failed"
        assert result.validity_status == "invalid"

    def test_no_scale_returns_degraded(self):
        """无标尺时应返回 degraded + pixel_only"""
        geom = CrackGeometryMeasurement()
        result = geom.run(
            inputs=[{'bbox_format': 'xywh', 'bbox': [100, 50, 100, 30]}],
            parameters={}, context={},
        )
        assert result.validity_status == "degraded"
        assert result.can_influence_decision is False
        assert result.structured_output["scale_available"] is False
        assert result.structured_output["length_mm"] is None

    def test_mask_uri_not_loaded(self):
        """mask_uri 引用但未加载文件时应警告"""
        geom = CrackGeometryMeasurement()
        result = geom.run(
            inputs=[{'segmentation_mask_uri': '/path/to/mask.npy'}],
            parameters={}, context={},
        )
        assert result.validity_status == "degraded"
        assert "MASK_URI_NOT_LOADED" in result.structured_output.get("warning", "")

    def test_no_format_rejected(self):
        """不提供 bbox_format 应被拒绝"""
        geom = CrackGeometryMeasurement()
        result = geom.run(
            inputs=[{'bbox': [100, 50, 200, 80]}],  # 没有 bbox_format
            parameters={}, context={},
        )
        assert result.execution_status == "failed"


class TestSeverityRater:
    """严重度分级测试"""

    def test_crack_severe(self):
        """2.8mm裂纹在HPT叶片上应为severe"""
        rater = SeverityRater()
        result = rater.run(
            inputs=[{'damage_type': 'crack',
                     'geometry': {'length_mm': 2.8},
                     'component': 'HPT Blade Stage 1'}],
            parameters={}, context={},
        )
        assert result.structured_output["severity"] == "severe"
        assert result.validity_status == "degraded"  # 通用阈值

    def test_area_mm2_not_confused_with_percent(self):
        """area_mm2 不应直接与百分比阈值比较"""
        rater = SeverityRater()
        result = rater.run(
            inputs=[{'damage_type': 'coating_spallation',
                     'geometry': {'area_mm2': 15},
                     'component': 'HPT Blade Stage 1'}],
            parameters={}, context={},
        )
        # 不应 crash；15mm² 不能直接与 30% 比较
        assert result.execution_status == "success"
        assert "area_percent" in str(result.structured_output.get("criteria_met", ""))


class TestDamageTypeClassifier:
    """损伤类型分类器测试"""

    def test_keyword_crack(self):
        """'dark linear indication' + crack → crack"""
        classifier = DamageTypeClassifier()
        result = classifier.run(
            inputs=[{'phenomenon': 'dark linear indication, possible crack'}],
            parameters={}, context={},
        )
        assert result.structured_output["damage_type"] == "crack"
        assert result.validity_status == "degraded"  # 关键词规则

    def test_unknown_phenomenon(self):
        """无法匹配的现象应返回 unknown"""
        classifier = DamageTypeClassifier()
        result = classifier.run(
            inputs=[{'phenomenon': 'unusual color variation'}],
            parameters={}, context={},
        )
        assert result.structured_output["damage_type"] == "unknown"
