"""官方资产注册引导 —— 将所有已调研验证的资产注册到 AssetRegistry。

使用方法:
    from aero_diag.registries import AssetRegistry
    from aero_diag.plugins.official.register import register_all_official_assets

    registry = AssetRegistry()
    register_all_official_assets(registry)

    # 按类别检索
    detectors = registry.search(kind=AssetKind.DETECTOR)
    reliability_models = registry.search(kind=AssetKind.RELIABILITY_MODEL)
"""

from __future__ import annotations

from aero_diag.assets.manifest import AssetKind, AssetStatus
from aero_diag.registries.asset_registry import AssetRegistry


def register_all_official_assets(registry: AssetRegistry) -> int:
    """将所有官方验证资产注册到注册中心。

    Returns:
        注册的资产总数
    """
    count = 0

    # ── 数据适配器 (5个) ──
    from .assets_inventory import (
        ASSET_CA2_DATASET,
        ASSET_BLADESYNTH_DATASET,
        ASSET_CMAPSS_DATASET,
        ASSET_TRUSTED_KE_ADAPTER,
        ASSET_BOEING_AVIATION_NER,
    )
    for m in [ASSET_CA2_DATASET, ASSET_BLADESYNTH_DATASET, ASSET_CMAPSS_DATASET,
              ASSET_TRUSTED_KE_ADAPTER, ASSET_BOEING_AVIATION_NER]:
        m.compute_digest()
        registry.register(m)
        count += 1

    # ── 预处理器 (3个) ──
    from .assets_inventory import (
        ASSET_PYVKF_ORDER_TRACKING,
        ASSET_SCIPY_SPECTRAL,
        ASSET_OPENCV_IMAGE_PREPROCESS,
    )
    for m in [ASSET_PYVKF_ORDER_TRACKING, ASSET_SCIPY_SPECTRAL, ASSET_OPENCV_IMAGE_PREPROCESS]:
        m.compute_digest()
        registry.register(m)
        count += 1

    # ── 检测器 (8个) ──
    from .assets_inventory import (
        ASSET_CA2_ANOMALY_DETECTOR,
        ASSET_EGCIENET_SEGMENTATION,
        ASSET_SLF_YOLO_METAL_DEFECT,
        ASSET_SAM_ADAPTER_CRACK,
        ASSET_TS_SAM_SEGMENTATION,
        ASSET_WCAMBA_BEARING_FAULT,
        ASSET_ISOLATION_FOREST_SCADA,
        ASSET_FAULTSENSE_LSTM_AUTOENCODER,
    )
    for m in [ASSET_CA2_ANOMALY_DETECTOR, ASSET_EGCIENET_SEGMENTATION,
              ASSET_SLF_YOLO_METAL_DEFECT, ASSET_SAM_ADAPTER_CRACK,
              ASSET_TS_SAM_SEGMENTATION, ASSET_WCAMBA_BEARING_FAULT,
              ASSET_ISOLATION_FOREST_SCADA, ASSET_FAULTSENSE_LSTM_AUTOENCODER]:
        m.compute_digest()
        registry.register(m)
        count += 1

    # ── 损伤表征工具 (3个) ──
    from .assets_inventory import (
        ASSET_CRACK_GEOMETRY_MEASUREMENT,
        ASSET_DAMAGE_CLASSIFIER,
        ASSET_SEVERITY_RATER,
    )
    for m in [ASSET_CRACK_GEOMETRY_MEASUREMENT, ASSET_DAMAGE_CLASSIFIER, ASSET_SEVERITY_RATER]:
        m.compute_digest()
        registry.register(m)
        count += 1

    # ── 可靠性/寿命模型 (6个) ──
    from .assets_inventory import (
        ASSET_CNN_LSTM_RUL,
        ASSET_PY_FATIGUE_PARIS,
        ASSET_FRAMEWORK_FDPP_PROBABILISTIC,
        ASSET_PYLIFE_SN_CURVE,
        ASSET_PINN_FLEET_PROGNOSIS,
        ASSET_CHANGEPOINT_LSTM_RUL,
    )
    for m in [ASSET_CNN_LSTM_RUL, ASSET_PY_FATIGUE_PARIS,
              ASSET_FRAMEWORK_FDPP_PROBABILISTIC, ASSET_PYLIFE_SN_CURVE,
              ASSET_PINN_FLEET_PROGNOSIS, ASSET_CHANGEPOINT_LSTM_RUL]:
        m.compute_digest()
        registry.register(m)
        count += 1

    # ── 知识源 (3个) ──
    from .assets_inventory import (
        ASSET_OMIN_KNOWLEDGE,
        ASSET_BOEING_KNOWLEDGE_NER,
        ASSET_MAINTIE_KNOWLEDGE,
    )
    for m in [ASSET_OMIN_KNOWLEDGE, ASSET_BOEING_KNOWLEDGE_NER, ASSET_MAINTIE_KNOWLEDGE]:
        m.compute_digest()
        registry.register(m)
        count += 1

    # ── 决策规则 (2个) ──
    from .assets_inventory import (
        ASSET_RISK_CLASSIFICATION_RULES,
        ASSET_INSPECTION_INTERVAL_RULES,
    )
    for m in [ASSET_RISK_CLASSIFICATION_RULES, ASSET_INSPECTION_INTERVAL_RULES]:
        m.compute_digest()
        registry.register(m)
        count += 1

    # ── 监控插件 (1个) ──
    from .assets_inventory import (
        ASSET_DATA_QUALITY_MONITOR,
    )
    ASSET_DATA_QUALITY_MONITOR.compute_digest()
    registry.register(ASSET_DATA_QUALITY_MONITOR)
    count += 1

    return count


def create_runner(registry: AssetRegistry | None = None) -> "AssetRunner":
    """创建可执行所有已注册资产的 AssetRunner。

    自动注册所有官方资产并构建实现表。
    """
    if registry is None:
        registry = AssetRegistry()
        register_all_official_assets(registry)

    from .asset_runner import AssetRunner, _build_default_impl_registry
    return AssetRunner(registry, _build_default_impl_registry())


__all__ = ["register_all_official_assets", "create_runner"]
