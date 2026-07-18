"""AssetRunner — 统一资产执行调度器。

职责：
1. 根据 asset_id 查找注册的资产 Manifest（适用域、风险、状态检查）
2. 查找对应的 Implementation 实现
3. 执行前置验证 → 运行 → 记录溯源
4. 优雅降级：未实现或有缺失依赖的资产返回清晰错误信息
"""

from __future__ import annotations

import time
from typing import Any

from aero_diag.assets.manifest import AssetStatus, EngineeringAssetManifest


class AssetNotAvailableError(Exception):
    """资产不可用。"""
    pass


class AssetRunner:
    """统一资产执行调度器。

    用法:
        runner = AssetRunner(registry, impl_registry)
        result = runner.execute("preprocessor.signal.scipy_spectral_analysis",
                                inputs=[{"signal": [...], "sample_rate": 1000}])
    """

    def __init__(self, registry: Any, impl_registry: dict[str, Any] | None = None):
        self._registry = registry  # AssetRegistry
        self._impl_registry = impl_registry or _build_default_impl_registry()

    def execute(
        self,
        asset_id: str,
        *,
        inputs: list[dict] | None = None,
        parameters: dict | None = None,
        context: dict | None = None,
        version: str = "latest",
    ) -> dict[str, Any]:
        """执行指定资产。

        Returns:
            {"status": "success"|"failed"|"unavailable",
             "manifest": {...}, "result": AssetRunResult,
             "error": str|None, "elapsed_ms": int}
        """
        inputs = inputs or []
        parameters = parameters or {}
        context = context or {}
        t0 = time.time()

        # 1. 查找资产
        try:
            entry = self._registry.resolve(asset_id, version)
        except Exception as e:
            return self._error(asset_id, f"Asset not found: {e}")

        manifest = entry.manifest

        # 2. 状态检查
        if manifest.status in (AssetStatus.DEPRECATED, AssetStatus.RETIRED):
            return self._error(asset_id, f"Asset is {manifest.status.value}")

        # 3. 查找实现
        impl = self._impl_registry.get(asset_id)
        if impl is None:
            return self._error(
                asset_id,
                f"No implementation registered. Available: {list(self._impl_registry.keys())}",
                manifest=manifest,
            )

        # 4. 前置验证
        validation = impl.validate_inputs(inputs, parameters, context)
        if not validation.get("ok", True):
            return {
                "status": "failed",
                "asset_id": asset_id,
                "manifest": _manifest_summary(manifest),
                "error": f"Input validation failed: {validation.get('issues', [])}",
                "elapsed_ms": int((time.time() - t0) * 1000),
            }

        # 5. 执行
        try:
            result = impl.run(inputs, parameters, context)
        except Exception as e:
            return {
                "status": "failed",
                "asset_id": asset_id,
                "manifest": _manifest_summary(manifest),
                "error": str(e),
                "elapsed_ms": int((time.time() - t0) * 1000),
            }

        elapsed_ms = result.elapsed_ms or int((time.time() - t0) * 1000)

        return {
            "status": result.status,
            "asset_id": asset_id,
            "manifest": _manifest_summary(manifest),
            "result": result,
            "elapsed_ms": elapsed_ms,
        }

    def is_available(self, asset_id: str) -> bool:
        """检查资产是否可用。"""
        try:
            entry = self._registry.resolve(asset_id)
            if entry.manifest.status in (AssetStatus.DEPRECATED, AssetStatus.RETIRED):
                return False
            return asset_id in self._impl_registry
        except Exception:
            return False

    def available_assets(self, kind: str | None = None) -> list[str]:
        """列出所有可执行资产。"""
        available = []
        for aid in self._impl_registry:
            try:
                entry = self._registry.resolve(aid)
                if kind and entry.manifest.asset_kind.value != kind:
                    continue
                if entry.manifest.status not in (AssetStatus.DEPRECATED, AssetStatus.RETIRED):
                    available.append(aid)
            except Exception:
                pass
        return available

    def _error(self, asset_id: str, msg: str, manifest=None) -> dict:
        return {
            "status": "unavailable",
            "asset_id": asset_id,
            "manifest": _manifest_summary(manifest) if manifest else None,
            "error": msg,
            "result": None,
            "elapsed_ms": 0,
        }


def _manifest_summary(m: EngineeringAssetManifest) -> dict:
    return {
        "asset_id": m.asset_id,
        "name": m.name,
        "version": m.version,
        "kind": m.asset_kind.value,
        "status": m.status.value,
        "intro_level": m.intro_level,
        "priority": m.priority,
        "risk": m.policy.risk,
        "requires_approval": m.policy.requires_approval,
    }


def _build_default_impl_registry() -> dict[str, Any]:
    """构建默认实现注册表——懒导入全部 31 个实现类。"""
    registry = {}

    def _try_register(cls):
        try:
            inst = cls()
            registry[cls.asset_id] = inst
        except Exception:
            pass

    # -- 预处理器 (3) --
    from .implementations.spectral import SpectralAnalysis; _try_register(SpectralAnalysis)
    from .implementations.opencv_preprocess import OpenCVPreprocessor; _try_register(OpenCVPreprocessor)
    from .implementations.pyvkf_adapter import PyVKFOrderTracking; _try_register(PyVKFOrderTracking)

    # -- 数据适配器 (5) --
    from .implementations.data_adapters import (
        CA2DatasetAdapter, CMAPSSDatasetAdapter, TrustedKEAdapter, BoeingNERAdapter,
    ); _try_register(CA2DatasetAdapter); _try_register(CMAPSSDatasetAdapter)
    _try_register(TrustedKEAdapter); _try_register(BoeingNERAdapter)
    from .implementations.pyvkf_adapter import BladeSynthAdapter; _try_register(BladeSynthAdapter)

    # -- 检测器 (8) --
    from .implementations.isolation_forest import IsolationForestDetector; _try_register(IsolationForestDetector)
    from .implementations.detectors_vision import (
        CA2AnomalyDetector, SLFYOLODetector, SAMAdapterCrackSegmentation,
        EGCIENetSegmentation, TSSAMSegmentation,
    ); _try_register(CA2AnomalyDetector); _try_register(SLFYOLODetector)
    _try_register(SAMAdapterCrackSegmentation); _try_register(EGCIENetSegmentation)
    _try_register(TSSAMSegmentation)
    from .implementations.detectors_signal import (
        WCambaBearingFaultDetector, FaultSenseLSTMDetector,
    ); _try_register(WCambaBearingFaultDetector); _try_register(FaultSenseLSTMDetector)

    # -- 表征工具 (3) --
    from .implementations.characterizers import (
        CrackGeometryMeasurement, DamageTypeClassifier, SeverityRater,
    ); _try_register(CrackGeometryMeasurement); _try_register(DamageTypeClassifier)
    _try_register(SeverityRater)

    # -- 可靠性/寿命 (6) --
    from .implementations.py_fatigue_runner import ParisLawCrackGrowth; _try_register(ParisLawCrackGrowth)
    from .implementations.rul_predictor import CNNLSTMRULPredictor; _try_register(CNNLSTMRULPredictor)
    from .implementations.reliability_extended import (
        FDPPProbabilisticCrackGrowth, PyLifeSNCurveCalculator,
        PinnFleetPrognosis, ChangePointLSTRUL,
    ); _try_register(FDPPProbabilisticCrackGrowth); _try_register(PyLifeSNCurveCalculator)
    _try_register(PinnFleetPrognosis); _try_register(ChangePointLSTRUL)

    # -- 监控 (1) --
    from .implementations.data_quality import DataQualityGate; _try_register(DataQualityGate)

    # -- 知识源 (3) --
    from .implementations.knowledge_and_rules import (
        OminKnowledgeSource, BoeingKnowledgeNER, MaintIEKnowledgeSource,
    ); _try_register(OminKnowledgeSource); _try_register(BoeingKnowledgeNER)
    _try_register(MaintIEKnowledgeSource)

    # -- 决策规则 (2) --
    from .implementations.knowledge_and_rules import (
        RiskClassificationRules, InspectionIntervalRules,
    ); _try_register(RiskClassificationRules); _try_register(InspectionIntervalRules)

    return registry
