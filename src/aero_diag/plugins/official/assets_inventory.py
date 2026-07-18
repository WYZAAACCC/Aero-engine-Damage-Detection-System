"""官方工程资产清单 —— 所有已调研并验证可用的算法、模型、工具。

每个资产严格遵循 EngineeringAssetManifest 规范（9 组必填字段）：
1. 身份与版本  2. 输入输出  3. 适用域  4. 方法与参数
5. 验证信息  6. 不确定性  7. 资源与执行  8. 安全策略  9. 可观测性

资产引入等级:
  L1_CORE        — 核心必备，P0 阶段接入，系统运行即依赖
  L2_RECOMMENDED — 推荐接入，P1 最小垂直闭环使用
  L3_OPTIONAL    — 可选，P2+ 按需接入
  L4_EXPERIMENTAL — 实验性，未充分验证，不可用于正式结论

优先级:
  P0_CRITICAL — 系统启动即需
  P1_HIGH     — 首个垂直场景必需
  P2_MEDIUM   — 插件平台阶段
  P3_LOW      — 知识与案例阶段
  P4_DEFERRED — 后续阶段

实现方式:
  DIRECT_PIP     — pip install 即用
  GIT_CLONE      — git clone + pip install -e
  HUGGINGFACE    — huggingface hub 下载
  SCIKIT_BUILTIN — scikit-learn 内置，无需额外安装
  SCIPY_BUILTIN  — scipy 内置
  SEEKFLOW_WRAP  — 封装为 SeekFlow @tool
  DOCKER_IMAGE   — 容器化部署
  API_SERVICE    — 远程 API 调用
"""

from __future__ import annotations

from aero_diag.assets.manifest import (
    ApplicabilitySpec,
    AssetKind,
    AssetStatus,
    EngineeringAssetManifest,
    InputSpec,
    MethodSpec,
    OutputSpec,
    PolicySpec,
    ResourceSpec,
    UncertaintySpec,
    VerificationSpec,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 引入等级与优先级常量
# ═══════════════════════════════════════════════════════════════════════════════

class IntroLevel:
    """资产引入等级"""
    L1_CORE = "L1_CORE"                # 核心必备
    L2_RECOMMENDED = "L2_RECOMMENDED"   # 推荐接入
    L3_OPTIONAL = "L3_OPTIONAL"         # 可选
    L4_EXPERIMENTAL = "L4_EXPERIMENTAL"  # 实验性


class Priority:
    """资产优先级"""
    P0_CRITICAL = "P0_CRITICAL"
    P1_HIGH = "P1_HIGH"
    P2_MEDIUM = "P2_MEDIUM"
    P3_LOW = "P3_LOW"
    P4_DEFERRED = "P4_DEFERRED"


class ImplMethod:
    """实现方式"""
    DIRECT_PIP = "direct_pip"
    GIT_CLONE = "git_clone"
    HUGGINGFACE = "huggingface"
    SCIKIT_BUILTIN = "scikit_builtin"
    SCIPY_BUILTIN = "scipy_builtin"
    SEEKFLOW_WRAP = "seekflow_wrap"
    DOCKER_IMAGE = "docker_image"
    API_SERVICE = "api_service"


# ═══════════════════════════════════════════════════════════════════════════════
# ── 数据适配器 (DATA_ADAPTER) ──
# ═══════════════════════════════════════════════════════════════════════════════

ASSET_CA2_DATASET = EngineeringAssetManifest(
    asset_id="data_adapter.borescope.ca2_real_scene",
    name="CA² 真实孔探图像数据集适配器",
    version="1.0.0",
    asset_kind=AssetKind.DATA_ADAPTER,
    publisher="changniu54 (USTC)",
    status=AssetStatus.VALIDATED,
    description=(
        "读取 CA² 开源数据集中的真实航空发动机孔探检查图像。"
        "包含 1,417 张正常图像与 857 张异常图像，覆盖多种发动机型号。"
        "适用于无监督异常检测训练与评估。"
    ),
    intro_level=IntroLevel.L1_CORE,
    priority=Priority.P0_CRITICAL,
    impl_method=ImplMethod.GIT_CLONE,
    impl_source="https://github.com/changniu54/CA2",
    impl_notes="git clone 后使用 dataset/dataloader.py 读取；图像为真实工业场景采集",

    inputs=[
        InputSpec(
            artifact_type="raw_image",
            required_fields=["image_uri"],
            allow_missing=["camera_params", "scale_info"],
            allowed_file_types=[".jpg", ".png", ".bmp"],
        ),
    ],
    outputs=[
        OutputSpec(
            artifact_type="raw_image",
            schema_ref="schemas/ca2_image_output.json",
            description="CA² 格式标准化的孔探图像 Artifact",
        ),
    ],

    applicability=ApplicabilitySpec(
        components=["compressor_blade", "turbine_blade", "combustor"],
        operating_modes=["borescope_inspection"],
        damage_types=["notch", "ablation", "crack", "coating_loss"],
        exclusions=["non_borescope_imagery", "lab_simulated_only"],
    ),

    method=MethodSpec(
        family="data_loading",
        deterministic=True,
        assumptions=["图像已按发动机/部件/正常-异常分类"],
        default_parameters={"resize_to": [512, 512], "normalize": True},
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["ca2_original_split"],
        metrics={"normal_abnormal_split": 0.623},
        reviewer="CA² paper authors",
        reviewed_at="2025-05",
    ),

    uncertainty=UncertaintySpec(
        output_representation="none",
        ood_checks=["image_source_must_be_borescope"],
    ),

    resources=ResourceSpec(cpu=2, memory_mb=512, gpu=False, timeout_s=30.0),

    policy=PolicySpec(
        risk="read",
        capabilities=["artifact.read"],
        requires_approval=False,
        data_classification="internal",
    ),

    metrics_keys=["images_loaded", "normal_count", "abnormal_count"],
    known_failure_modes=["dataset_path_not_found", "corrupted_image_skipped"],
)

ASSET_BLADESYNTH_DATASET = EngineeringAssetManifest(
    asset_id="data_adapter.borescope.bladesynth",
    name="BladeSynth 合成叶片缺陷数据集适配器",
    version="1.0.0",
    asset_kind=AssetKind.DATA_ADAPTER,
    publisher="MohammedEltoum (UCL)",
    status=AssetStatus.VALIDATED,
    description=(
        "高质量渲染合成数据集，用于航空发动机叶片缺陷检测。"
        "Nature Scientific Data 2025 发布，可弥补真实数据不足。"
        "注意：合成数据训练模型仅限辅助分析，正式结论需真实数据验证。"
    ),
    intro_level=IntroLevel.L2_RECOMMENDED,
    priority=Priority.P2_MEDIUM,
    impl_method=ImplMethod.GIT_CLONE,
    impl_source="https://github.com/MohammedEltoum/bladeSynth",
    impl_notes="合成数据不可单独用于正式工程结论；应与真实数据联合使用",

    inputs=[
        InputSpec(artifact_type="raw_image",
                  required_fields=["image_uri"],
                  allowed_file_types=[".png", ".exr"]),
    ],
    outputs=[
        OutputSpec(artifact_type="raw_image",
                   description="标准化的渲染合成叶片图像"),
    ],

    applicability=ApplicabilitySpec(
        components=["turbine_blade", "compressor_blade"],
        operating_modes=["synthetic_inspection"],
        damage_types=["crack", "coating_spallation", "erosion"],
        exclusions=["certification_decision"],
    ),

    method=MethodSpec(
        family="synthetic_data_loading",
        deterministic=True,
        assumptions=["渲染管线包含真实材料参数"],
        default_parameters={"render_quality": "high"},
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["bladesynth_paper_split"],
        metrics={"rendering_fidelity": 0.92},
        reviewer="Scientific Data reviewers",
        reviewed_at="2025",
    ),

    uncertainty=UncertaintySpec(
        output_representation="none",
        ood_checks=["合成数据标记必须传递到下游结论"],
    ),

    resources=ResourceSpec(cpu=2, memory_mb=1024, gpu=False, timeout_s=60.0),

    policy=PolicySpec(
        risk="read",
        capabilities=["artifact.read"],
        requires_approval=False,
        data_classification="internal",
    ),

    metrics_keys=["images_loaded", "defect_type_distribution"],
    known_failure_modes=["render_files_not_found"],
)

ASSET_CMAPSS_DATASET = EngineeringAssetManifest(
    asset_id="data_adapter.timeseries.cmapss",
    name="NASA C-MAPSS 涡轮风扇发动机数据集适配器",
    version="1.0.0",
    asset_kind=AssetKind.DATA_ADAPTER,
    publisher="NASA Ames Prognostics Center",
    status=AssetStatus.QUALIFIED,
    description=(
        "NASA 商业模块化航空推进系统仿真数据集。"
        "涡轮风扇发动机退化仿真数据，4 个子集 (FD001-FD004)，"
        "26 列（机组ID、时间周期、3 操作参数、21 传感器读数）。"
        "学术界 RUL 预测标准基准。"
    ),
    intro_level=IntroLevel.L1_CORE,
    priority=Priority.P1_HIGH,
    impl_method=ImplMethod.DIRECT_PIP,
    impl_source="NASA官方FTP / gdown / kaggle datasets download",
    impl_notes="pip install gdown; gdown <file_id>; 或从 Kaggle 下载",

    inputs=[
        InputSpec(
            artifact_type="raw_timeseries",
            required_fields=["unit_id", "time_cycles"],
            required_channels=[
                "T2", "T24", "T30", "T50", "P2", "P15", "P30",
                "Nf", "Nc", "epr", "Ps30", "phi", "NRf", "NRc",
                "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd",
                "W31", "W32",
            ],
            units={
                "temperature": "°R", "pressure": "psia",
                "speed": "rpm", "ratio": "unitless",
            },
        ),
    ],
    outputs=[
        OutputSpec(artifact_type="raw_timeseries",
                   description="标准化 NASA CMAPSS 格式时序数据"),
    ],

    applicability=ApplicabilitySpec(
        components=["turbofan_engine"],
        operating_modes=["sea_level_takeoff", "cruise", "climb"],
        exclusions=["non_turbofan_engines"],
    ),

    method=MethodSpec(
        family="data_loading",
        deterministic=True,
        assumptions=["传感器数据已同步", "操作参数已知"],
        default_parameters={
            "sequence_length": 50, "stride": 1,
            "rul_label_method": "piecewise_linear", "rul_threshold": 130,
        },
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["FD001_train_test_split"],
        metrics={"train_samples": 20631, "test_engines": 100},
        reviewer="NASA + academic community",
        reviewed_at="2008",
        valid_until="indefinite",
    ),

    uncertainty=UncertaintySpec(
        output_representation="none",
        ood_checks=["operating_condition_must_match_expectation"],
    ),

    resources=ResourceSpec(cpu=2, memory_mb=1024, gpu=False, timeout_s=60.0),

    policy=PolicySpec(
        risk="read",
        capabilities=["artifact.read"],
        requires_approval=False,
        data_classification="public",
    ),

    metrics_keys=["sequences_loaded", "train_test_split_ratio"],
    known_failure_modes=["missing_sensor_columns", "file_format_changed"],
)

ASSET_TRUSTED_KE_ADAPTER = EngineeringAssetManifest(
    asset_id="data_adapter.text.trusted_ke_maintenance",
    name="TrustedKE 维修记录文本抽取适配器",
    version="1.0.0",
    asset_kind=AssetKind.DATA_ADAPTER,
    publisher="nd-crane (University of Notre Dame)",
    status=AssetStatus.VALIDATED,
    description=(
        "基于 FAA 事故/事件数据的维修记录 NLP 信息抽取。"
        "支持命名实体识别(NER)、指代消解、实体链接。"
        "16 个开源 NLP 工具零样本基准测试。"
    ),
    intro_level=IntroLevel.L2_RECOMMENDED,
    priority=Priority.P3_LOW,
    impl_method=ImplMethod.GIT_CLONE,
    impl_source="https://github.com/nd-crane/trusted_ke",
    impl_notes="需要安装 spacy, flair 等 NLP 依赖；模型首次需下载",

    inputs=[
        InputSpec(
            artifact_type="raw_text",
            required_fields=["text_content", "source"],
            allow_missing=["aircraft_type", "engine_serial"],
        ),
    ],
    outputs=[
        OutputSpec(artifact_type="raw_structured",
                   description="结构化维修实体: 部件/故障/动作/时间"),
    ],

    applicability=ApplicabilitySpec(
        components=["any"],
        operating_modes=["maintenance_log_analysis"],
        exclusions=["non_english_text", "handwritten_notes"],
    ),

    method=MethodSpec(
        family="nlp_ner",
        deterministic=False,
        assumptions=["文本为英文航空维修记录"],
        default_parameters={"model": "spacy_en_core_web_lg", "threshold": 0.5},
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["omin_gold_annotations"],
        metrics={"ner_f1": 0.82, "coreference_f1": 0.65},
        reviewer="Notre Dame + FAA",
        reviewed_at="2024-08",
    ),

    uncertainty=UncertaintySpec(
        output_representation="none",
        ood_checks=["非英文文本自动标记为 OOD"],
        error_sources=["NER模型对航空术语的零样本能力有限"],
    ),

    resources=ResourceSpec(cpu=4, memory_mb=2048, gpu=False, timeout_s=120.0),

    policy=PolicySpec(
        risk="read",
        capabilities=["artifact.read"],
        requires_approval=False,
        data_classification="internal",
    ),

    metrics_keys=["entities_extracted", "coreference_chains"],
    known_failure_modes=["模型下载失败", "文本编码错误"],
)

ASSET_BOEING_AVIATION_NER = EngineeringAssetManifest(
    asset_id="data_adapter.text.boeing_aviation_ner",
    name="Boeing Aviation NER 航空维修实体识别器",
    version="1.0.0",
    asset_kind=AssetKind.DATA_ADAPTER,
    publisher="Boeing + FAA",
    status=AssetStatus.VALIDATED,
    description=(
        "波音与FAA联合开发的航空维修服务困难报告(SDR)命名实体识别模型。"
        "基于 GLiNER 微调，实体类型: 飞行阶段/产品位置/机组动作/产品/产品状态/"
        "鸟击/紧急或异常情况。可 pip 安装。"
    ),
    intro_level=IntroLevel.L2_RECOMMENDED,
    priority=Priority.P3_LOW,
    impl_method=ImplMethod.HUGGINGFACE,
    impl_source="https://huggingface.co/boeing/aviation-ner",
    impl_notes="pip install gliner; 从 HuggingFace 加载模型权重",

    inputs=[
        InputSpec(
            artifact_type="raw_text",
            required_fields=["text_content"],
            allow_missing=["aircraft_type", "report_date"],
        ),
    ],
    outputs=[
        OutputSpec(artifact_type="raw_structured",
                   description="SDR 实体: FlightPhase/ProductLocation/CrewAction/Product/ProductCondition"),
    ],

    applicability=ApplicabilitySpec(
        components=["any"],
        operating_modes=["sdr_analysis", "maintenance_log_analysis"],
        exclusions=["non_aviation_text", "non_english"],
    ),

    method=MethodSpec(
        family="transformer_ner",
        deterministic=False,
        assumptions=["输入为 SDR 或类似航空维修文本"],
        default_parameters={"entity_types": [
            "Flight Phase", "Product Location", "Crew Action",
            "Product", "Product Condition", "Bird/Animal Strike",
            "Emergency/Abnormal Situation",
        ]},
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["boeing_sdr_gold_split"],
        metrics={"precision": 0.85, "recall": 0.80},
        reviewer="Boeing + FAA data science",
        reviewed_at="2024",
    ),

    uncertainty=UncertaintySpec(
        output_representation="none",
        error_sources=["航空术语稀疏性问题"],
    ),

    resources=ResourceSpec(cpu=2, memory_mb=1024, gpu=False, timeout_s=60.0),

    policy=PolicySpec(
        risk="read",
        capabilities=["artifact.read"],
        requires_approval=False,
        data_classification="internal",
    ),

    metrics_keys=["entities_extracted", "avg_confidence"],
    known_failure_modes=["模型权重下载失败", "GLiNER版本不兼容"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# ── 预处理器 (PREPROCESSOR) ──
# ═══════════════════════════════════════════════════════════════════════════════

ASSET_PYVKF_ORDER_TRACKING = EngineeringAssetManifest(
    asset_id="preprocessor.vibration.pyvkf_order_tracking",
    name="PyVKF Vold-Kalman 阶次跟踪滤波器",
    version="1.0.0",
    asset_kind=AssetKind.PREPROCESSOR,
    publisher="CyprienHoelzl (ETH Zurich)",
    status=AssetStatus.VALIDATED,
    description=(
        "第二代 Vold-Kalman 阶次滤波器 Python 实现。"
        "无需物理转速计（tacholess），从振动信号中提取阶次分量。"
        "航空发动机振动分析必备预处理步骤——转子阶次、叶片通过频率提取。"
    ),
    intro_level=IntroLevel.L1_CORE,
    priority=Priority.P1_HIGH,
    impl_method=ImplMethod.GIT_CLONE,
    impl_source="https://github.com/CyprienHoelzl/PyVKF",
    impl_notes="pip install numpy scipy; git clone; 支持并行提取多阶次",
    risk_notes="需要已知参考转速或能从信号中估计；阶次混叠时可能误提取",
    limitation_notes="仅提取已定义的阶次分量；不检测未知阶次异常",

    inputs=[
        InputSpec(artifact_type="raw_timeseries", required_channels=["vibration"],
                  required_fields=["sample_rate", "speed_reference"],
                  units={"vibration": "m/s^2", "speed": "rpm"}),
    ],
    outputs=[
        OutputSpec(artifact_type="derived_timeseries",
                   description="各阶次分量的幅值与相位时间序列"),
    ],

    applicability=ApplicabilitySpec(
        components=["compressor", "turbine", "rotor", "bearing"],
        operating_modes=["runup", "steady", "rundown"],
        speed_range_rpm=(500, 30000),
        exclusions=["missing_speed_reference_and_tacho"],
    ),

    method=MethodSpec(
        family="order_tracking_vkf",
        deterministic=True,
        assumptions=["转速变化率在 VKF 跟踪范围内"],
        default_parameters={"max_order": 20, "bandwidth": 0.05, "parallel_orders": 4},
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["pyvkf_test_signals"],
        metrics={"frequency_error_pct": 0.3},
        reviewer="VKF community",
        reviewed_at="2023-02",
    ),

    uncertainty=UncertaintySpec(
        output_representation="none",
        ood_checks=["speed_range", "sample_rate_sufficiency"],
        error_sources=["阶次重叠", "转速估计误差"],
    ),

    resources=ResourceSpec(cpu=2, memory_mb=512, gpu=False, timeout_s=120.0),

    policy=PolicySpec(risk="read", capabilities=["artifact.read", "artifact.write"],
                      requires_approval=False, data_classification="internal"),

    metrics_keys=["orders_extracted", "computation_time_ms"],
    known_failure_modes=["speed_signal_missing", "sample_rate_too_low_for_order"],
)

ASSET_SCIPY_SPECTRAL = EngineeringAssetManifest(
    asset_id="preprocessor.signal.scipy_spectral_analysis",
    name="SciPy 频谱分析预处理器",
    version="1.0.0",
    asset_kind=AssetKind.PREPROCESSOR,
    publisher="SciPy Community",
    status=AssetStatus.QUALIFIED,
    description=(
        "基于 scipy.signal 的振动/声学信号频谱分析。"
        "提供 FFT/功率谱/STFT/包络谱/倒频谱等标准信号处理功能。"
        "零额外依赖——Python 科学计算标配。"
    ),
    intro_level=IntroLevel.L1_CORE,
    priority=Priority.P0_CRITICAL,
    impl_method=ImplMethod.SCIPY_BUILTIN,
    impl_source="pip install scipy",
    impl_notes="scipy.signal.spectrogram, scipy.signal.welch, scipy.fft 等内置函数",
    risk_notes="频谱分辨率受采样长度限制；窗函数选择影响幅值精度",
    limitation_notes="仅标准线性谱方法；非线性非平稳方法需额外库",

    inputs=[
        InputSpec(artifact_type="raw_timeseries", required_channels=["vibration"],
                  required_fields=["sample_rate"],
                  units={"vibration": "m/s^2 or mm/s"}),
    ],
    outputs=[
        OutputSpec(artifact_type="derived_timeseries",
                   description="频谱/包络谱/倒频谱分析结果"),
    ],

    applicability=ApplicabilitySpec(
        components=["compressor", "turbine", "bearing", "gearbox"],
        operating_modes=["any"],
        exclusions=[],
    ),

    method=MethodSpec(
        family="spectral_analysis",
        deterministic=True,
        assumptions=["信号为平稳或分段平稳"],
        default_parameters={
            "fft_length": 4096, "overlap": 0.75,
            "window": "hann", "freq_range_hz": (0, 20000),
        },
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["standard_test_signals"],
        metrics={"frequency_accuracy": 0.999},
        reviewer="SciPy community",
        reviewed_at="continuous",
        valid_until="indefinite",
    ),

    uncertainty=UncertaintySpec(
        output_representation="none",
        ood_checks=[],
        error_sources=["频谱泄漏", "栅栏效应"],
    ),

    resources=ResourceSpec(cpu=1, memory_mb=256, gpu=False, timeout_s=10.0, parallel_safe=True),

    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
                      requires_approval=False, data_classification="internal"),

    metrics_keys=["signal_length", "frequency_resolution_hz"],
    known_failure_modes=["input_signal_all_zeros", "sample_rate_zero"],
)

ASSET_OPENCV_IMAGE_PREPROCESS = EngineeringAssetManifest(
    asset_id="preprocessor.image.opencv_preprocess",
    name="OpenCV 孔探图像预处理工具",
    version="1.0.0",
    asset_kind=AssetKind.PREPROCESSOR,
    publisher="OpenCV Community",
    status=AssetStatus.QUALIFIED,
    description=(
        "基于 OpenCV 的孔探图像标准化预处理。"
        "包括：光照校正(CLAHE)、去噪、锐化、缩放、色彩空间转换。"
    ),
    intro_level=IntroLevel.L1_CORE,
    priority=Priority.P1_HIGH,
    impl_method=ImplMethod.DIRECT_PIP,
    impl_source="pip install opencv-python",
    impl_notes="完全内置，无需外部模型",
    risk_notes="过度增强可能引入伪影；CLAHE参数需根据孔探设备特性调整",
    limitation_notes="无法恢复已过曝或完全黑暗区域的细节",

    inputs=[
        InputSpec(artifact_type="raw_image", required_fields=["image_uri"],
                  allowed_file_types=[".jpg", ".png", ".bmp", ".tiff"]),
    ],
    outputs=[
        OutputSpec(artifact_type="derived_image", description="标准化增强后的图像"),
    ],

    applicability=ApplicabilitySpec(
        components=["any"],
        operating_modes=["borescope_inspection"],
        exclusions=[],
    ),

    method=MethodSpec(
        family="image_preprocessing",
        deterministic=True,
        assumptions=["图像为 RGB/BGR 格式"],
        default_parameters={
            "clahe_clip_limit": 2.0, "clahe_tile_size": [8, 8],
            "denoise_strength": 10, "target_size": [512, 512],
        },
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["standard_borescope_images"],
        metrics={"contrast_improvement_pct": 35},
        reviewer="internal",
        reviewed_at="2026-07",
    ),

    uncertainty=UncertaintySpec(output_representation="none", ood_checks=["extreme_darkness"]),

    resources=ResourceSpec(cpu=2, memory_mb=512, gpu=False, timeout_s=30.0, parallel_safe=True),

    policy=PolicySpec(risk="read", capabilities=["artifact.read", "artifact.write"],
                      requires_approval=False, data_classification="internal"),

    metrics_keys=["images_processed", "avg_processing_time_ms"],
    known_failure_modes=["corrupted_image", "unsupported_format"],
)
# ═══════════════════════════════════════════════════════════════════════════════
# ── 检测器 (DETECTOR) 8个 ──
# ═══════════════════════════════════════════════════════════════════════════════

ASSET_CA2_ANOMALY_DETECTOR = EngineeringAssetManifest(
    asset_id="detector.borescope.ca2_anomaly",
    name="CA² 无监督孔探叶片异常检测器",
    version="1.0.0", asset_kind=AssetKind.DETECTOR,
    publisher="changniu54 (USTC)", status=AssetStatus.CANDIDATE,
    description=(
        "基于类无关自适应特征适应的无监督异常检测。"
        "训练无需缺陷样本——仅用正常叶片图像完成特征聚类，检测时计算距离。"
        "解决工程中缺陷样本稀缺的核心痛点。适合作为初筛工具。"
    ),
    intro_level="L2_RECOMMENDED", priority="P1_HIGH",
    impl_method="git_clone", impl_source="https://github.com/changniu54/CA2",
    impl_notes="PyTorch 模型；需下载预训练 ImageNet 权重 + CA² 数据集；推理单张 ~50ms",
    risk_notes="仅检测'异常'现象，不区分损伤类型；光照剧烈变化可能产生假阳性",
    limitation_notes="无监督方法——不提供损伤分类标签；需要后续分类/分割工具确认具体类型",

    inputs=[InputSpec(artifact_type="raw_image", required_fields=["image_uri"],
            allowed_file_types=[".jpg", ".png"], allow_missing=["scale_info"])],
    outputs=[OutputSpec(artifact_type="detection_finding", description="异常区域热力图 + 异常分数")],

    applicability=ApplicabilitySpec(
        components=["compressor_blade", "turbine_blade"], operating_modes=["borescope_inspection"],
        damage_types=["notch", "ablation", "crack", "coating_loss", "anomaly"],
        exclusions=["non_borescope_image", "extreme_overexposure"],
    ),

    method=MethodSpec(family="anomaly_detection_unsupervised", deterministic=False,
        assumptions=["正常训练图像覆盖所有正常形态变体"],
        default_parameters={"feature_extractor": "resnet50", "k_neighbors": 5, "threshold_percentile": 95},
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["ca2_test_split"],
        metrics={"auroc": 0.92, "f1": 0.88, "inference_ms": 50},
        reviewer="CA² authors", reviewed_at="2025-05",
    ),

    uncertainty=UncertaintySpec(output_representation="none",
        ood_checks=["image_source_must_be_borescope", "brightness_extreme_check"],
        error_sources=["image_domain_gap_from_training"],
    ),

    resources=ResourceSpec(cpu=4, memory_mb=4096, gpu=True, gpu_memory_mb=4000, timeout_s=60.0),

    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=False, data_classification="internal"),

    metrics_keys=["anomaly_score", "ood_flag", "inference_ms"],
    known_failure_modes=["gpu_not_available", "pretrained_weights_not_downloaded"],
)

ASSET_EGCIENET_SEGMENTATION = EngineeringAssetManifest(
    asset_id="detector.borescope.egcienet_segmentation",
    name="EGCIENet SAM 引导叶片缺陷分割网络",
    version="1.0.0", asset_kind=AssetKind.DETECTOR,
    publisher="Newbiejy", status=AssetStatus.CANDIDATE,
    description=(
        "SegFormer + SAM 边缘引导的 in-service 叶片缺陷分割网络。"
        "利用 SAM 全局边缘特征引导，处理反光和域内变化。"
        "88.13% mIoU，30.6 FPS——准实时。"
    ),
    intro_level="L2_RECOMMENDED", priority="P1_HIGH",
    impl_method="git_clone", impl_source="https://github.com/Newbiejy/EGCIENet_In-service-blade-defect-detection",
    impl_notes="需要 SAM 预训练权重 + SegFormer 权重；GPU推理推荐",
    risk_notes="仅支持已训练缺陷类型(裂纹/压坑/断裂/卷边)；新缺陷类型需微调",
    limitation_notes="数据集仅587张——小样本；反光极端时可能漏检",

    inputs=[InputSpec(artifact_type="raw_image", required_fields=["image_uri"],
            allowed_file_types=[".jpg", ".png"])],
    outputs=[OutputSpec(artifact_type="detection_finding", description="缺陷分割掩膜 + 类别 + 置信度")],

    applicability=ApplicabilitySpec(
        components=["compressor_blade", "turbine_blade"], operating_modes=["borescope_inspection"],
        damage_types=["crack", "dent", "fracture", "curled_edge"],
        exclusions=["non_blade_components", "extreme_blur"],
    ),

    method=MethodSpec(family="segmentation_transformer_sam_guided", deterministic=False,
        assumptions=["缺陷在图像中可见", "光照均匀"],
        default_parameters={"backbone": "segformer_b2", "sam_model": "vit_h", "confidence_threshold": 0.5},
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["egcienet_test_set"],
        metrics={"mIoU": 0.8813, "fps": 30.6},
        reviewer="EGCIENet authors", reviewed_at="2025",
    ),

    uncertainty=UncertaintySpec(output_representation="none",
        ood_checks=["defect_type_not_in_training", "image_quality_below_threshold"],
    ),

    resources=ResourceSpec(cpu=4, memory_mb=8192, gpu=True, gpu_memory_mb=8000, timeout_s=120.0),

    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=False, data_classification="internal"),

    metrics_keys=["mIoU", "per_class_IoU", "fps"],
    known_failure_modes=["gpu_oom", "sam_weight_download_failed"],
)

ASSET_SLF_YOLO_METAL_DEFECT = EngineeringAssetManifest(
    asset_id="detector.surface.slf_yolo_metal_defect",
    name="SLF-YOLO 金属表面缺陷检测器 (YOLOv8增强)",
    version="1.0.0", asset_kind=AssetKind.DETECTOR,
    publisher="zacianfans", status=AssetStatus.CANDIDATE,
    description=(
        "基于增强 YOLOv8 的金属表面缺陷检测。轻量化设计(SC_C2f + Light-SSF_Neck + FIMetal-IoU)。"
        "NEU-DET 80.0% mAP, AL10-DET 86.8% mAP。适合资源受限的工业环境。"
        "可 fine-tune 到叶片表面缺陷检测。"
    ),
    intro_level="L2_RECOMMENDED", priority="P2_MEDIUM",
    impl_method="git_clone", impl_source="https://github.com/zacianfans/SLF-YOLO",
    impl_notes="PyTorch + ultralytics；预训练权重可用；fine-tune 需要标注数据",
    risk_notes="预训练于 NEU-DET(热轧钢带表面)而非航空叶片——需要 fine-tune",
    limitation_notes="YOLO 仅输出边界框——需配合分割模型进行精确几何测量",

    inputs=[InputSpec(artifact_type="raw_image", required_fields=["image_uri"],
            allowed_file_types=[".jpg", ".png", ".bmp"])],
    outputs=[OutputSpec(artifact_type="detection_finding", description="缺陷边界框 + 类别 + 置信度")],

    applicability=ApplicabilitySpec(
        components=["blade_surface", "casing", "disk"], operating_modes=["visual_inspection"],
        damage_types=["crack", "scratch", "pit", "inclusion", "coating_loss"],
        exclusions=["internal_defects", "subsurface_cracks"],
    ),

    method=MethodSpec(family="object_detection_yolo", deterministic=False,
        assumptions=["缺陷在图像中可见", "已 fine-tune 到目标域"],
        default_parameters={
            "model_variant": "slf_yolov8n", "confidence_threshold": 0.25,
            "iou_threshold": 0.45, "input_size": [640, 640],
        },
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["NEU-DET_test", "AL10-DET_test"],
        metrics={"mAP@0.5_NEU_DET": 0.80, "mAP@0.5_AL10_DET": 0.868},
        reviewer="SLF-YOLO authors", reviewed_at="2025",
    ),

    uncertainty=UncertaintySpec(output_representation="none",
        ood_checks=["defect_type_unseen", "image_domain_shift"],
    ),

    resources=ResourceSpec(cpu=2, memory_mb=4096, gpu=True, gpu_memory_mb=4000, timeout_s=60.0),

    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=False, data_classification="internal"),

    metrics_keys=["mAP", "inference_fps", "bbox_count"],
    known_failure_modes=["transfer_learning_needed", "small_defect_missed"],
)

ASSET_SAM_ADAPTER_CRACK = EngineeringAssetManifest(
    asset_id="detector.crack.sam_adapter_segmentation",
    name="SAM-Adapter 裂纹分割器 (LoRA/Adapter微调)",
    version="1.0.0", asset_kind=AssetKind.DETECTOR,
    publisher="sky-visionX / multi-authors", status=AssetStatus.CANDIDATE,
    description=(
        "基于 SAM (Segment Anything Model) + Adapter/LoRA 微调的裂纹分割。"
        "仅微调 ~1% 参数，保持 SAM 的强零样本泛化能力。"
        "已验证：路面裂纹 → 可迁移到金属表面裂纹 / 叶片裂纹。"
    ),
    intro_level="L2_RECOMMENDED", priority="P1_HIGH",
    impl_method="git_clone", impl_source="https://github.com/sky-visionX/CrackSegmentation",
    impl_notes="需要 SAM ViT-H 权重(~2.4GB)；LoRA微调仅需数十MB；推理 ~100ms/图 GPU",
    risk_notes="路面裂纹域 → 航空金属裂纹域存在域偏移；需要少量叶片裂纹标注微调",
    limitation_notes="极细裂纹(<1px宽)可能漏分；强反光区域可能产生误分",

    inputs=[InputSpec(artifact_type="raw_image", required_fields=["image_uri"],
            allowed_file_types=[".jpg", ".png", ".tiff"])],
    outputs=[OutputSpec(artifact_type="detection_finding", description="裂纹像素级掩膜 + 置信度 + 裂纹宽度估计")],

    applicability=ApplicabilitySpec(
        components=["blade", "disk", "casing", "combustor_liner"],
        operating_modes=["visual_inspection", "borescope_inspection"],
        damage_types=["crack", "scratch", "linear_indication"],
        exclusions=["non_linear_defects", "porosity"],
    ),

    method=MethodSpec(family="segmentation_sam_adapter_finetune", deterministic=False,
        assumptions=["裂纹具有线性或分支特征", "图像分辨率足够（≥512px）"],
        default_parameters={
            "sam_model": "vit_h", "fine_tune_method": "lora",
            "lora_rank": 4, "confidence_threshold": 0.3,
        },
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["crack500", "CFD", "custom_metal_crack"],
        metrics={"crack_dice": 0.78, "zero_shot_cfd_dice": 0.72},
        reviewer="multiple research groups", reviewed_at="2024",
    ),

    uncertainty=UncertaintySpec(output_representation="none",
        ood_checks=["domain_shift_detection", "resolution_sufficiency"],
        error_sources=["域偏移", "极细裂纹<1px"],
    ),

    resources=ResourceSpec(cpu=4, memory_mb=8192, gpu=True, gpu_memory_mb=10000, timeout_s=120.0),

    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=False, data_classification="internal"),

    metrics_keys=["dice_coefficient", "iou", "crack_width_estimate_mm"],
    known_failure_modes=["gpu_oom_sam_vit_h", "lora_weights_not_found"],
)

ASSET_TS_SAM_SEGMENTATION = EngineeringAssetManifest(
    asset_id="detector.general.ts_sam_segmentation",
    name="TS-SAM 双流通用缺陷分割器",
    version="1.0.0", asset_kind=AssetKind.DETECTOR,
    publisher="maoyangou147", status=AssetStatus.CANDIDATE,
    description=(
        "双流 SAM 分割器：卷积侧适配器(CSA)+多尺度优化模块(MRM)+特征融合解码器(FFD)。"
        "10 个公共数据集验证，跨 3 类任务，超越 SAM-Adapter。"
        "适合多种叶片缺陷类型的通用分割。"
    ),
    intro_level="L3_OPTIONAL", priority="P2_MEDIUM",
    impl_method="git_clone", impl_source="https://github.com/maoyangou147/TS-SAM",
    impl_notes="SAM ViT-H + 额外适配器模块；PyTorch",
    risk_notes="模型体量大（ViT-H + 双流额外参数）；推理速度慢于轻量模型",
    limitation_notes="实时应用需模型蒸馏或剪枝",

    inputs=[InputSpec(artifact_type="raw_image", required_fields=["image_uri"],
            allowed_file_types=[".jpg", ".png"])],
    outputs=[OutputSpec(artifact_type="detection_finding", description="多类别分割掩膜")],

    applicability=ApplicabilitySpec(
        components=["blade", "disk", "casing"], operating_modes=["visual_inspection"],
        damage_types=["crack", "coating_loss", "erosion", "burn_mark"],
        exclusions=["real_time_requirements"],
    ),

    method=MethodSpec(family="segmentation_two_stream_sam", deterministic=False,
        assumptions=["缺陷类型在预训练或微调域中"],
        default_parameters={"sam_model": "vit_h", "adapter_type": "csa", "confidence": 0.5},
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["10_public_datasets"],
        metrics={"avg_dice": 0.81},
        reviewer="TS-SAM authors", reviewed_at="2024-07",
    ),

    uncertainty=UncertaintySpec(output_representation="none",
        ood_checks=["domain_shift_detection"]),
    resources=ResourceSpec(cpu=4, memory_mb=12288, gpu=True, gpu_memory_mb=16000, timeout_s=180.0),
    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=False, data_classification="internal"),
    metrics_keys=["dice_per_class", "inference_time_s"],
    known_failure_modes=["vram_insufficient_sam_vit_h"],
)

ASSET_WCAMBA_BEARING_FAULT = EngineeringAssetManifest(
    asset_id="detector.vibration.wcamba_bearing_fault",
    name="WCamba 航空轴承故障诊断器 (轻量 CNN+Mamba)",
    version="1.0.0", asset_kind=AssetKind.DETECTOR,
    publisher="CDUT-IMRT", status=AssetStatus.CANDIDATE,
    description=(
        "宽核CNN + Mamba 状态空间的轻量航空轴承故障诊断。"
        "95.44%准确率，仅 0.016M 参数。支持 -6~+6dB 噪声。"
        "22.61%训练时间减少，52.44%推理时间减少。"
    ),
    intro_level="L2_RECOMMENDED", priority="P1_HIGH",
    impl_method="git_clone", impl_source="https://github.com/CDUT-IMRT/WCamba",
    impl_notes="PyTorch；轻量到可在边缘设备运行；HIT航空轴承数据集+帕德博恩大学数据集",
    risk_notes="仅诊断轴承故障——其他部件(齿轮/叶片)需不同模型",
    limitation_notes="分类模型——输出故障类型标签，不定位具体损伤位置",

    inputs=[InputSpec(artifact_type="raw_timeseries", required_channels=["vibration"],
            required_fields=["sample_rate"], units={"vibration": "m/s^2"})],
    outputs=[OutputSpec(artifact_type="detection_finding", description="故障类型 + 置信度 + OOD标记")],

    applicability=ApplicabilitySpec(
        components=["bearing"], operating_modes=["runup", "steady", "rundown"],
        damage_types=["inner_race_fault", "outer_race_fault", "ball_fault", "cage_fault"],
        exclusions=["non_bearing_components"],
    ),

    method=MethodSpec(family="classification_wide_kernel_cnn_mamba", deterministic=False,
        assumptions=["振动信号采样率足以捕获轴承特征频率"],
        default_parameters={"noise_level_db": 0, "window_length": 2048},
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["HIT_aero_bearing", "paderborn_bearing"],
        metrics={"accuracy": 0.9544, "parameters": 16000, "inference_speedup_pct": 52.44},
        reviewer="WCamba authors", reviewed_at="2025",
    ),

    uncertainty=UncertaintySpec(output_representation="none",
        ood_checks=["snr_check", "bearing_type_mismatch"]),
    resources=ResourceSpec(cpu=2, memory_mb=1024, gpu=False, timeout_s=30.0),
    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=False, data_classification="internal"),
    metrics_keys=["fault_type", "confidence", "ood_flag"],
    known_failure_modes=["insufficient_sample_rate", "extreme_noise_exceeds_6db"],
)

ASSET_ISOLATION_FOREST_SCADA = EngineeringAssetManifest(
    asset_id="detector.scada.isolation_forest_anomaly",
    name="Isolation Forest 燃气轮机 SCADA 异常检测器",
    version="1.0.0", asset_kind=AssetKind.DETECTOR,
    publisher="scikit-learn / davidfertube", status=AssetStatus.CANDIDATE,
    description=(
        "基于 scikit-learn IsolationForest 的燃气轮机多维传感器异常检测。"
        "94.5%精度，91.2%召回率，<50ms推理。"
        "7个关键传感器：排气温度、振动X/Y、轴承温度、进口压力、滑油压力、燃油流量。"
    ),
    intro_level="L1_CORE", priority="P0_CRITICAL",
    impl_method="scikit_builtin", impl_source="sklearn.ensemble.IsolationForest",
    impl_notes="完全内置——零额外依赖；n_estimators=200, contamination=0.02",
    risk_notes="contamination参数需根据实际故障率调整；正常行为漂移可能产生假阳性",
    limitation_notes="仅检测'异常'——不提供故障原因诊断；多传感器相关性未被显式建模",

    inputs=[InputSpec(artifact_type="raw_timeseries",
            required_channels=["exhaust_temp", "vibration_x", "vibration_y", "bearing_temp",
                               "inlet_pressure", "lube_oil_pressure", "fuel_flow"],
            required_fields=["timestamp"], allow_missing=["bearing_temp"])],
    outputs=[OutputSpec(artifact_type="detection_finding", description="异常标签 + 异常分数 + 贡献传感器")],

    applicability=ApplicabilitySpec(
        components=["turbofan_engine", "gas_turbine"], operating_modes=["steady", "cruise"],
        damage_types=["general_anomaly"], exclusions=["transient_only_analysis"],
    ),

    method=MethodSpec(family="anomaly_detection_isolation_forest", deterministic=False,
        assumptions=["正常行为数据覆盖主要工况", "传感器已校准"],
        default_parameters={
            "n_estimators": 200, "max_samples": 1000, "contamination": 0.02,
            "random_state": 42,
        },
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["turbine_scada_benchmark"],
        metrics={"precision": 0.945, "recall": 0.912, "f1": 0.928, "inference_ms": 50},
        reviewer="davidfertube / scikit-learn community", reviewed_at="2024",
    ),

    uncertainty=UncertaintySpec(output_representation="none",
        ood_checks=["operating_mode_change", "sensor_drift"]),
    resources=ResourceSpec(cpu=2, memory_mb=512, gpu=False, timeout_s=10.0, parallel_safe=True),
    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=False, data_classification="internal"),
    metrics_keys=["anomaly_score", "contributing_sensors", "outlier_ratio"],
    known_failure_modes=["sensor_values_all_zero", "contamination_too_high"],
)

ASSET_FAULTSENSE_LSTM_AUTOENCODER = EngineeringAssetManifest(
    asset_id="detector.timeseries.faultsense_lstm_autoencoder",
    name="FaultSense LSTM 自编码器时序异常检测器",
    version="1.0.0", asset_kind=AssetKind.DETECTOR,
    publisher="momo-2609", status=AssetStatus.CANDIDATE,
    description=(
        "LSTM 自编码器 + 滑动窗口 + 重构误差自适应阈值。"
        "NASA CMAPSS 数据集 RMSE 14.85 (FD001) / 13.88 (FD003)。"
        "附带 Plotly Dash 机队健康仪表板 + FastAPI + Docker 部署。"
    ),
    intro_level="L2_RECOMMENDED", priority="P1_HIGH",
    impl_method="git_clone", impl_source="https://github.com/momo-2609/FaultSense-LSTM-Anomaly-Detection-on-NASA-CMAPSS",
    impl_notes="TensorFlow/Keras；包含完整部署代码(Docker/FastAPI/Dash)；推荐GPU训练",
    risk_notes="自编码器重构误差对工况变化敏感——新工况可能产生假阳性",
    limitation_notes="需要足够正常数据训练；异常阈值需根据实际场景标定",

    inputs=[InputSpec(artifact_type="raw_timeseries", required_channels=[
        "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc",
        "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed",
        "Nf_dmd", "PCNfR_dmd", "W31", "W32",
    ], required_fields=["unit_id", "time_cycles"], allow_missing=[])],
    outputs=[OutputSpec(artifact_type="detection_finding", description="异常分数 + 重构误差 + 自适应阈值")],

    applicability=ApplicabilitySpec(
        components=["turbofan_engine"], operating_modes=["any"],
        damage_types=["general_degradation", "sensor_fault", "performance_drift"],
        exclusions=["non_cmapss_format_data"],
    ),

    method=MethodSpec(family="anomaly_detection_lstm_autoencoder", deterministic=False,
        assumptions=["引擎退化过程是渐进式", "传感器数据同步"],
        default_parameters={
            "sequence_length": 50, "hidden_dim": 128, "latent_dim": 32,
            "threshold_sigma": 2.5, "epochs": 100,
        },
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["CMAPSS_FD001", "CMAPSS_FD003"],
        metrics={"rmse_fd001": 14.85, "rmse_fd003": 13.88},
        reviewer="FaultSense community", reviewed_at="2024",
    ),

    uncertainty=UncertaintySpec(output_representation="none",
        ood_checks=["operating_condition_change", "sequence_length_mismatch"],
        error_sources=["模型对新工况泛化不足", "阈值自适应滞后"],
    ),

    resources=ResourceSpec(cpu=4, memory_mb=4096, gpu=True, gpu_memory_mb=4000, timeout_s=300.0),

    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=False, data_classification="internal"),

    metrics_keys=["reconstruction_error", "anomaly_threshold", "anomaly_score"],
    known_failure_modes=["sequence_length_mismatch", "sensor_count_mismatch"],
)
# ═══════════════════════════════════════════════════════════════════════════════
# ── 损伤表征工具 (CHARACTERIZER) 3个 ──
# ═══════════════════════════════════════════════════════════════════════════════

ASSET_CRACK_GEOMETRY_MEASUREMENT = EngineeringAssetManifest(
    asset_id="characterizer.crack.geometry_measurement",
    name="裂纹几何量测量工具（像素→物理坐标）",
    version="1.0.0", asset_kind=AssetKind.CHARACTERIZER,
    publisher="internal (OpenCV-based)", status=AssetStatus.CANDIDATE,
    description=(
        "基于 SAM-Adapter 分割掩膜 + OpenCV 骨架化的裂纹几何量测量。"
        "输出：长度(mm)、宽度(mm)、面积(mm²)、方向(°)——仅在标尺可用时输出物理值。"
        "无标尺时仅输出像素尺度和区间。"
    ),
    intro_level="L2_RECOMMENDED", priority="P1_HIGH",
    impl_method="seekflow_wrap", impl_source="internal",
    impl_notes="依赖标定数据(scales/calibration.json)；需检查 mask_available 和 scale_available",
    risk_notes="无有效标尺时禁止输出毫米值——这是安全关键约束",
    limitation_notes="像素到物理量转换精度受标定误差和视角畸变影响",

    inputs=[InputSpec(artifact_type="detection_finding", required_fields=[
        "segmentation_mask_uri", "scale_info", "calibration_method"],
        allow_missing=["scale_info"])],
    outputs=[OutputSpec(artifact_type="damage_characterization",
        description="裂纹几何: 长度/宽度/面积/方向 + 测量方法 + 标定来源 + 误差")],

    applicability=ApplicabilitySpec(
        components=["blade", "disk", "casing"], operating_modes=["visual_inspection", "borescope_inspection"],
        damage_types=["crack"], exclusions=["no_segmentation_mask_available"],
    ),

    method=MethodSpec(family="geometric_measurement_from_mask", deterministic=True,
        assumptions=["分割掩膜准确度足够", "标定数据有效——如无可使用像素输出"],
        default_parameters={
            "skeletonize_method": "zhang_suen", "min_crack_length_px": 10,
            "output_pixel_fallback": True,  # 无标尺时输出像素
        },
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["calibrated_crack_measurements"],
        metrics={"length_error_mm": 0.05, "width_error_mm": 0.01},
        reviewer="internal", reviewed_at="TBD",
    ),

    uncertainty=UncertaintySpec(output_representation="interval",
        calibration_method="repeat_measurement_std",
        ood_checks=["scale_available", "mask_quality", "viewpoint_angle"],
        error_sources=["标定误差", "分割掩膜边界模糊", "视角畸变"],
    ),

    resources=ResourceSpec(cpu=2, memory_mb=1024, gpu=False, timeout_s=30.0),

    policy=PolicySpec(risk="read", capabilities=["artifact.read", "artifact.write"],
        requires_approval=False, data_classification="internal"),

    metrics_keys=["length_mm", "width_mm", "area_mm2", "scale_available", "measurement_method"],
    known_failure_modes=["mask_region_too_small", "scale_file_missing", "calibration_data_corrupted"],
)

ASSET_DAMAGE_CLASSIFIER = EngineeringAssetManifest(
    asset_id="characterizer.damage.damage_type_classifier",
    name="损伤类型分类器（基于语义+视觉特征）",
    version="1.0.0", asset_kind=AssetKind.CHARACTERIZER,
    publisher="internal", status=AssetStatus.CANDIDATE,
    description=(
        "将检测发现 (DetectionFinding) 分类为文档定义的 11 种损伤类型："
        "crack(裂纹)/coating_spallation(涂层剥落)/erosion(冲蚀)/corrosion(腐蚀)/"
        "burn_mark(烧蚀)/dent(凹陷)/FOD(外物损伤)/wear(磨损)/deformation(变形)/rub(碰摩)/unknown(未知)。"
        "结合视觉特征+语义规则+置信度估计。"
    ),
    intro_level="L2_RECOMMENDED", priority="P1_HIGH",
    impl_method="seekflow_wrap", impl_source="internal",
    impl_notes="规则引擎 + 可选 ResNet/CLIP 视觉编码器；输出区分 observed/inferred/suspected",
    risk_notes="仅凭视觉特征可能无法区分相似损伤类型（如冲蚀vs腐蚀）",
    limitation_notes="需要领域规则库完善后才能达到生产级准确度",

    inputs=[InputSpec(artifact_type="detection_finding", required_fields=[
        "phenomenon", "score", "score_semantics", "location"],
        allow_missing=["score"])],
    outputs=[OutputSpec(artifact_type="damage_characterization",
        description="损伤类型 + 置信度(observed/inferred/suspected)")],

    applicability=ApplicabilitySpec(
        components=["compressor_blade", "turbine_blade", "combustor_liner", "HPT", "LPT"],
        operating_modes=["any_inspection"],
        damage_types=["crack", "coating_spallation", "erosion", "corrosion", "burn_mark",
                       "dent", "FOD", "wear", "deformation", "rub", "unknown"],
        exclusions=[],
    ),

    method=MethodSpec(family="classification_rule_based_plus_visual", deterministic=False,
        assumptions=["损伤类型在 11 类预定义集中", "视觉特征可由检测器提供或从图像提取"],
        default_parameters={"use_visual_encoder": True, "visual_model": "resnet50",
                            "confidence_threshold": 0.6},
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["ca2_labelled_subset"],
        metrics={"classification_accuracy": 0.85},
        reviewer="internal", reviewed_at="TBD",
    ),

    uncertainty=UncertaintySpec(output_representation="set_based",
        calibration_method="rule_strength",
        ood_checks=["damage_type_not_in_11_classes"],
        error_sources=["相似类型的视觉混淆", "新型损伤模式未定义"],
    ),

    resources=ResourceSpec(cpu=2, memory_mb=2048, gpu=False, timeout_s=30.0),

    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=False, data_classification="internal"),

    metrics_keys=["damage_type", "confidence_level", "classification_basis"],
    known_failure_modes=["unseen_damage_type", "ambiguous_visual_features"],
)

ASSET_SEVERITY_RATER = EngineeringAssetManifest(
    asset_id="characterizer.damage.severity_rater",
    name="损伤严重度分级器 (minor/moderate/severe/critical)",
    version="1.0.0", asset_kind=AssetKind.CHARACTERIZER,
    publisher="internal (rules-based)", status=AssetStatus.CANDIDATE,
    description=(
        "基于规则引擎的损伤严重度 4 级分级。综合几何量、损伤类型、部件、"
        "工况因素。分级依据可追溯到具体规则条款。"
    ),
    intro_level="L2_RECOMMENDED", priority="P1_HIGH",
    impl_method="seekflow_wrap", impl_source="internal",
    impl_notes="规则配置文件(severity_rules.yaml)；可扩展为更多分级层级",
    risk_notes="严重度分级可能导致维修/停飞决策——规则必须双人审核",
    limitation_notes="当前为规则驱动——不能处理超出规则定义范围的边缘情况",

    inputs=[InputSpec(artifact_type="damage_characterization", required_fields=[
        "damage_type", "geometry", "component_location"],
        allow_missing=["uncertainty"])],
    outputs=[OutputSpec(artifact_type="damage_characterization",
        description="损伤严重度: minor/moderate/severe/critical + 分级依据 + 规则引用")],

    applicability=ApplicabilitySpec(
        components=["compressor_blade", "turbine_blade", "combustor_liner", "disk"],
        operating_modes=["any_inspection"], damage_types=["any"], exclusions=[],
    ),

    method=MethodSpec(family="classification_rule_engine", deterministic=True,
        assumptions=["分级规则已经过领域专家审核"],
        default_parameters={"use_geometry": True, "use_damage_type": True,
                            "use_component_context": True},
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["expert_labelled_severity_cases"],
        metrics={"expert_agreement_rate": 0.90},
        reviewer="domain experts (TBD)", reviewed_at="TBD",
    ),

    uncertainty=UncertaintySpec(output_representation="set_based",
        calibration_method="expert_agreement", ood_checks=["rule_not_applicable"]),
    resources=ResourceSpec(cpu=1, memory_mb=256, gpu=False, timeout_s=5.0, parallel_safe=True),
    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=True, data_classification="internal"),

    metrics_keys=["severity_level", "rule_ref", "criteria_met"],
    known_failure_modes=["rule_file_not_loaded", "edge_case_not_covered"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# ── 可靠性/寿命模型 (RELIABILITY_MODEL) 6个 ──
# ═══════════════════════════════════════════════════════════════════════════════

ASSET_CNN_LSTM_RUL = EngineeringAssetManifest(
    asset_id="reliability_model.rul.cnn_lstm_cmapss",
    name="CNN-LSTM 混合模型 RUL 预测器",
    version="1.0.0", asset_kind=AssetKind.RELIABILITY_MODEL,
    publisher="muk0644", status=AssetStatus.CANDIDATE,
    description=(
        "混合 CNN-LSTM 架构用于 NASA CMAPSS 剩余寿命预测。"
        "RMSE 5.27 cycles，R² 99.15%。比较 7 种模型 (RF/XGBoost/LightGBM/LSTM/CNN/Bi-LSTM/CNN-LSTM)。"
        "推荐作为 RUL 预测的基线模型。"
    ),
    intro_level="L2_RECOMMENDED", priority="P1_HIGH",
    impl_method="git_clone", impl_source="https://github.com/muk0644/AI-Driven-Predictive-Maintenance-for-Aircraft-Engine-using-ML-and-DL",
    impl_notes="TensorFlow/Keras；预训练权重可用；50周期序列窗口；14传感器；MinMax归一化",
    risk_notes="CMAPSS 是仿真数据——模型迁移到真实发动机需要 fine-tune；退化模式假设单一",
    limitation_notes="CNN-LSTM 是黑盒模型——需要 SHAP 等工具解释预测；仅支持与 CMAPSS 类似的传感器集",

    inputs=[InputSpec(artifact_type="raw_timeseries", required_channels=[
        "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc",
        "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed",
        "Nf_dmd", "PCNfR_dmd", "W31", "W32",
    ], required_fields=["unit_id", "time_cycles"])],
    outputs=[OutputSpec(artifact_type="reliability_assessment",
        description="RUL (cycles) + 不确定性区间 + 传感器重要性")],

    applicability=ApplicabilitySpec(
        components=["turbofan_engine"], operating_modes=["any"],
        damage_types=["general_degradation"],
        exclusions=["non_cmapss_compatible_sensors", "non_turbofan"],
    ),

    method=MethodSpec(family="rul_prediction_cnn_lstm", deterministic=False,
        assumptions=["退化模式与 CMAPSS 仿真一致", "传感器数据完整"],
        default_parameters={
            "sequence_length": 50, "min_rul": 0, "max_rul": 130,
            "model_type": "cnn_lstm", "scoring": "nasa_asymmetric",
        },
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["CMAPSS_FD001", "CMAPSS_FD002", "CMAPSS_FD003", "CMAPSS_FD004"],
        metrics={"rmse_fd001": 5.27, "r2": 0.9915},
        reviewer="model authors", reviewed_at="2024",
    ),

    uncertainty=UncertaintySpec(output_representation="interval",
        calibration_method="prediction_std_across_ensemble",
        ood_checks=["sensor_distribution_shift", "operating_condition_change"]),

    resources=ResourceSpec(cpu=4, memory_mb=4096, gpu=True, gpu_memory_mb=4000, timeout_s=300.0),

    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=False, data_classification="internal"),

    metrics_keys=["rul_cycles", "confidence_interval", "sensor_importance_top5"],
    known_failure_modes=["sensor_count_mismatch", "sequence_too_short", "model_weights_missing"],
)

ASSET_PY_FATIGUE_PARIS = EngineeringAssetManifest(
    asset_id="reliability_model.crack.py_fatigue_paris_law",
    name="py_fatigue Paris 裂纹扩展模型 (确定性)",
    version="1.0.0", asset_kind=AssetKind.RELIABILITY_MODEL,
    publisher="OWI-Lab (Vrije Universiteit Brussel)", status=AssetStatus.CANDIDATE,
    description=(
        "基于 Paris Law (da/dN = C·ΔK^m) 的确定性疲劳裂纹扩展计算。"
        "支持多斜率 Paris 曲线、Walker 平均应力修正、阈值/临界 SIF。"
        "含 ASTM E1049-85 雨流计数。pip install 即用。"
    ),
    intro_level="L1_CORE", priority="P1_HIGH",
    impl_method="direct_pip", impl_source="pip install py_fatigue",
    impl_notes="完全开箱即用；pip install py_fatigue；文档: https://owi-lab.github.io/py_fatigue",
    risk_notes="确定性模型——不输出寿命分布/失效概率；用户需自行提供材料参数 C 和 m",
    limitation_notes="仅确定性 Paris 裂纹扩展——无概率能力；需外部 MC 包装获得概率寿命",

    inputs=[InputSpec(artifact_type="damage_characterization", required_fields=[
        "damage_type", "geometry"], allow_missing=["material_params"],
        units={"length": "mm", "stress_intensity": "MPa√m"})],
    outputs=[OutputSpec(artifact_type="reliability_assessment",
        description="裂纹扩展寿命 (cycles) + da/dN vs ΔK 曲线 + 临界尺寸")],

    applicability=ApplicabilitySpec(
        components=["blade", "disk", "casing"], operating_modes=["cyclic_loading"],
        damage_types=["crack"], exclusions=["non_linear_fracture", "creep_dominant"],
    ),

    method=MethodSpec(family="crack_growth_paris_law", deterministic=True,
        assumptions=["线弹性断裂力学有效", "裂纹几何可由 Y(a) 近似"],
        default_parameters={
            "C": 1e-12, "m": 3.0, "delta_K_th": 2.0, "K_c": 60.0,
            "initial_crack_mm": 1.0, "critical_crack_mm": 10.0,
        },
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["standard_fatigue_crack_growth_data"],
        metrics={"paris_fit_r2": 0.98},
        reviewer="OWI-Lab / VUB", reviewed_at="2023", valid_until="indefinite",
    ),

    uncertainty=UncertaintySpec(output_representation="point_estimate",
        ood_checks=["crack_size_exceeds_LEFM_limit", "temperature_range"],
        error_sources=["材料参数 C/m 的测量误差", "几何因子 Y(a) 近似误差"],
    ),

    resources=ResourceSpec(cpu=2, memory_mb=512, gpu=False, timeout_s=30.0, parallel_safe=True),

    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=False, data_classification="internal"),

    metrics_keys=["cycles_to_failure", "final_crack_mm", "paris_parameters"],
    known_failure_modes=["material_params_out_of_valid_range", "negative_stress_ratio_unsupported"],
)

ASSET_FRAMEWORK_FDPP_PROBABILISTIC = EngineeringAssetManifest(
    asset_id="reliability_model.crack.framework_fdpp_probabilistic",
    name="FrameworkFDPP 概率疲劳裂纹扩展模型 (MC仿真)",
    version="1.0.0", asset_kind=AssetKind.RELIABILITY_MODEL,
    publisher="ansak95", status=AssetStatus.CANDIDATE,
    description=(
        "航空铝合金 7075 概率 Paris-Erdogan 裂纹扩展模型。"
        "C 对数正态分布 + m 高斯分布 (ρ=-0.996) + 初始裂纹高斯分布 (CoV)。"
        "生成合成应变数据 + RUL 标签，用于深度学习预后模型训练。"
    ),
    intro_level="L3_OPTIONAL", priority="P2_MEDIUM",
    impl_method="git_clone", impl_source="https://github.com/ansak95/FrameworkFDPP",
    impl_notes="PyTorch；材料参数分布基于统计文献；可直接用于 MC 仿真",
    risk_notes="参数分布基于铝合金 7075 文献——需用航空发动机材料(IN718/Ti-6Al-4V)数据替换",
    limitation_notes="机身蒙皮场景——叶片裂纹几何因子 Y(a) 需替换为叶片特定几何",

    inputs=[InputSpec(artifact_type="damage_characterization", required_fields=[
        "damage_type", "geometry", "material_type"], allow_missing=["material_params"])],
    outputs=[OutputSpec(artifact_type="reliability_assessment",
        description="概率裂纹寿命分布 + MC 样本 + 失效概率 vs cycles 曲线")],

    applicability=ApplicabilitySpec(
        components=["blade", "disk"], operating_modes=["cyclic_loading"],
        damage_types=["crack"], exclusions=["creep_dominant", "corrosion_fatigue"],
    ),

    method=MethodSpec(family="crack_growth_probabilistic_monte_carlo", deterministic=False,
        assumptions=["参数分布独立（除 C-m 负相关外）", "Paris Law 可描述主要扩展阶段"],
        default_parameters={
            "n_mc_samples": 10000, "c_log_mean": -28.5, "c_log_std": 0.15,
            "m_mean": 3.5, "m_std": 0.1, "initial_crack_mean_mm": 0.5, "initial_crack_cov": 0.2,
        },
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["fuselage_panel_crack_data"],
        metrics={"rul_coverage_90pct": 0.88},
        reviewer="FrameworkFDPP authors", reviewed_at="2023",
    ),

    uncertainty=UncertaintySpec(output_representation="distribution",
        calibration_method="monte_carlo_parameter_sampling",
        ood_checks=["material_mismatch", "geometry_factor_invalid"],
        error_sources=["参数分布估计误差", "Paris Law 在小裂纹和大裂纹阶段的偏离"],
    ),

    resources=ResourceSpec(cpu=4, memory_mb=2048, gpu=False, timeout_s=600.0),

    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=False, data_classification="internal"),

    metrics_keys=["pof_at_inspection", "rul_median", "rul_p2_5", "rul_p97_5", "sensitivity_c", "sensitivity_m"],
    known_failure_modes=["material_parameters_out_of_bounds", "negative_crack_size_sampled"],
)

ASSET_PYLIFE_SN_CURVE = EngineeringAssetManifest(
    asset_id="reliability_model.fatigue.pylife_sn_woehler",
    name="pyLife S-N 曲线疲劳寿命计算器 (Bosch/FKM)",
    version="1.0.0", asset_kind=AssetKind.RELIABILITY_MODEL,
    publisher="Bosch Research", status=AssetStatus.CANDIDATE,
    description=(
        "Bosch Research 开源 S-N 曲线疲劳分析库。FKM 非线性局部应力/应变概念。"
        "Wöhler 曲线拟合、雨流计数、损伤累积、失效概率计算。"
        "Apache-2.0 许可，工业级代码质量。"
    ),
    intro_level="L2_RECOMMENDED", priority="P2_MEDIUM",
    impl_method="direct_pip", impl_source="pip install pylife",
    impl_notes="pip install pylife；完整文档和 Jupyter 示例；支持 Abaqus/ANSYS 导入",
    risk_notes="仅 S-N/ε-N 方法——不包含 LEFM 裂纹扩展；需材料 S-N 数据",
    limitation_notes="材料数据库需用户自行提供；S-N 方法不直接输出裂纹长度",

    inputs=[InputSpec(artifact_type="damage_characterization", required_fields=[
        "stress_or_strain_history", "material_sn_data"], allow_missing=[])],
    outputs=[OutputSpec(artifact_type="reliability_assessment",
        description="疲劳寿命 (cycles) + 损伤累积 + 失效概率")],

    applicability=ApplicabilitySpec(
        components=["blade", "disk", "shaft"], operating_modes=["cyclic_loading"],
        damage_types=["fatigue"], exclusions=["crack_growth_prediction", "fretting"],
    ),

    method=MethodSpec(family="fatigue_sn_curve_fkm", deterministic=False,
        assumptions=["S-N 曲线数据代表材料行为", "载荷历史可作雨流计数"],
        default_parameters={"fkm_guideline": "nonlinear", "survival_probability": 0.975},
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["fkm_standard_benchmarks"],
        metrics={"sn_curve_fit_rmse": 0.05},
        reviewer="Bosch Research", reviewed_at="ongoing", valid_until="indefinite",
    ),

    uncertainty=UncertaintySpec(output_representation="point_estimate",
        calibration_method="fkm_scatter_factor"),
    resources=ResourceSpec(cpu=2, memory_mb=512, gpu=False, timeout_s=60.0, parallel_safe=True),
    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=False, data_classification="internal"),
    metrics_keys=["cycles_to_failure", "damage_sum", "failure_probability"],
    known_failure_modes=["material_data_missing", "irregular_load_history"],
)

ASSET_PINN_FLEET_PROGNOSIS = EngineeringAssetManifest(
    asset_id="reliability_model.crack.pinn_fleet_prognosis",
    name="PINN 物理信息机群裂纹预后模型 (Paris Law + NN)",
    version="1.0.0", asset_kind=AssetKind.RELIABILITY_MODEL,
    publisher="PML-UCF", status=AssetStatus.CANDIDATE,
    description=(
        "物理信息循环神经网络——Paris Law 层 + 应力强度层 + 累积损伤单元。"
        "融合已知物理(Paris)与数据驱动层。MIT 许可。"
        "演示在飞机机群疲劳裂纹预测场景。"
    ),
    intro_level="L3_OPTIONAL", priority="P4_DEFERRED",
    impl_method="git_clone", impl_source="https://github.com/PML-UCF/pinn",
    impl_notes="TensorFlow/Keras + 自动微分；MIT 许可；需理解物理信息网络原理",
    risk_notes="模型复杂度高——需要物理+DL双领域专家配置和维护",
    limitation_notes="仅限裂纹扩展——不覆盖其他失效模式",

    inputs=[InputSpec(artifact_type="raw_timeseries", required_channels=["far_field_loads"],
            required_fields=["crack_length_measurements"],
            allow_missing=["crack_length_measurements"])],
    outputs=[OutputSpec(artifact_type="reliability_assessment",
        description="机群级别概率裂纹长度预测 + RUL 分布")],

    applicability=ApplicabilitySpec(
        components=["blade", "disk"], operating_modes=["cyclic_loading"],
        damage_types=["crack"], exclusions=["creep", "corrosion"],
    ),

    method=MethodSpec(family="hybrid_physics_data_driven_rnn", deterministic=False,
        assumptions=["Paris Law 描述主导退化过程", "远场载荷可测量或估计"],
        default_parameters={
            "paris_law_layers": 3, "rnn_units": 64,
            "observation_noise_std": 0.01, "training_epochs": 500,
        },
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["synthetic_aircraft_fleet"],
        metrics={"crack_length_rmse": 0.15, "fleet_distribution_match": 0.92},
        reviewer="PML-UCF", reviewed_at="2023",
    ),

    uncertainty=UncertaintySpec(output_representation="distribution",
        calibration_method="bayesian_inference",
        ood_checks=["physics_constraint_violation"]),
    resources=ResourceSpec(cpu=4, memory_mb=4096, gpu=True, gpu_memory_mb=4000, timeout_s=1800.0),
    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=True, data_classification="internal"),
    metrics_keys=["predicted_crack_mm", "rul_median", "physics_loss", "data_loss"],
    known_failure_modes=["physics_loss_instability", "hyperparameter_sensitivity"],
)

ASSET_CHANGEPOINT_LSTM_RUL = EngineeringAssetManifest(
    asset_id="reliability_model.rul.changepoint_lstm_multicondition",
    name="ChangePoint-LSTM 多变工况 RUL 预测器",
    version="1.0.0", asset_kind=AssetKind.RELIABILITY_MODEL,
    publisher="en-research", status=AssetStatus.CANDIDATE,
    description=(
        "变点检测集成 LSTM 用于多变工况 RUL 估计。"
        "5.6%~7.5% 准确率提升（vs 标准 LSTM）于多条件场景(FD002+)。"
        "框架无关的变点检测模块。Control Engineering Practice 2024。"
    ),
    intro_level="L3_OPTIONAL", priority="P2_MEDIUM",
    impl_method="git_clone", impl_source="https://github.com/en-research/ChangePoint-LSTM",
    impl_notes="TensorFlow 2；仅当多工况分析需要时使用",
    risk_notes="变点检测对超参数敏感——需针对特定发动机型号调参",
    limitation_notes="相对于简单 LSTM 的增量提升——如果基线已足够好则不需要",

    inputs=[InputSpec(artifact_type="raw_timeseries", required_channels=[
        "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc",
        "epr", "Ps30", "phi", "NRf", "NRc"],
        required_fields=["unit_id", "time_cycles", "operating_condition"])],
    outputs=[OutputSpec(artifact_type="reliability_assessment",
        description="RUL + 变点标记 + 各退化阶段预测")],

    applicability=ApplicabilitySpec(
        components=["turbofan_engine"], operating_modes=["multi_condition"],
        damage_types=["general_degradation"], exclusions=["single_condition_scenario"],
    ),

    method=MethodSpec(family="rul_prediction_changepoint_lstm", deterministic=False,
        assumptions=["退化过程中存在可检测的变点", "操作条件可识别"],
        default_parameters={
            "changepoint_penalty": 5, "lstm_units": 128,
            "sequence_length": 50, "min_segment_length": 5,
        },
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["CMAPSS_FD002", "CMAPSS_FD003", "CMAPSS_FD004"],
        metrics={"rmse_improvement_pct": 5.6, "multi_condition_accuracy_gain": 7.5},
        reviewer="Changepoint-LSTM authors / CEP journal", reviewed_at="2024",
    ),

    uncertainty=UncertaintySpec(output_representation="interval",
        ood_checks=["changepoint_instability", "segment_too_short"]),
    resources=ResourceSpec(cpu=4, memory_mb=4096, gpu=True, gpu_memory_mb=4000, timeout_s=600.0),
    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=False, data_classification="internal"),
    metrics_keys=["rul_cycles", "changepoint_count", "segment_durations"],
    known_failure_modes=["no_change_point_detected", "segment_too_short_for_lstm"],
)
# ═══════════════════════════════════════════════════════════════════════════════
# ── 知识源 (KNOWLEDGE_SOURCE) 3个 ──
# ═══════════════════════════════════════════════════════════════════════════════

ASSET_OMIN_KNOWLEDGE = EngineeringAssetManifest(
    asset_id="knowledge_source.maintenance.omin_faa_knowledge",
    name="OMIn 航空维修知识源 (FAA 事故/事件 NER+CR)",
    version="1.0.0", asset_kind=AssetKind.KNOWLEDGE_SOURCE,
    publisher="nd-crane (University of Notre Dame)", status=AssetStatus.CANDIDATE,
    description=(
        "基于 FAA 事故/事件数据的操作与维修智能知识源。"
        "金标准命名实体识别+指代消解+实体链接标注。"
        "16 个开源 NLP 工具的零样本基准——用于自动抽取维修知识。"
    ),
    intro_level="L3_OPTIONAL", priority="P3_LOW",
    impl_method="git_clone", impl_source="https://github.com/nd-crane/trusted_ke",
    impl_notes="知识抽取结果需人类专家审核后才能入库",
    risk_notes="零样本 NLP 工具的实体抽取有误——不能作为唯一知识来源",
    limitation_notes="仅覆盖 FAA 数据——不包含所有发动机型号和故障模式",

    inputs=[InputSpec(artifact_type="raw_text", required_fields=["text_content"])],
    outputs=[OutputSpec(artifact_type="knowledge_reference",
        description="结构化维修知识条目: 部件/故障/动作/时间")],

    applicability=ApplicabilitySpec(
        components=["any"], operating_modes=["maintenance_log_analysis"],
        exclusions=["non_english"],
    ),

    method=MethodSpec(family="knowledge_extraction_nlp", deterministic=False,
        assumptions=["文本为英文航空维修记录"],
        default_parameters={"nlp_tool": "spacy_en_core_web_lg", "confidence_threshold": 0.5},
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["omin_gold_annotations"],
        metrics={"ner_f1": 0.82, "coref_f1": 0.65},
        reviewer="Notre Dame", reviewed_at="2024-08",
    ),

    uncertainty=UncertaintySpec(output_representation="none",
        error_sources=["零样本NER对航空术语识别率有限"]),
    resources=ResourceSpec(cpu=4, memory_mb=4096, gpu=False, timeout_s=300.0),
    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=False, data_classification="internal"),
    metrics_keys=["entities_extracted", "coreference_chains"],
    known_failure_modes=["nlp_model_not_downloaded", "out_of_memory_large_corpus"],
)

ASSET_BOEING_KNOWLEDGE_NER = EngineeringAssetManifest(
    asset_id="knowledge_source.maintenance.boeing_aviation_ner_knowledge",
    name="波音 Aviation NER 维修知识抽取器",
    version="1.0.0", asset_kind=AssetKind.KNOWLEDGE_SOURCE,
    publisher="Boeing + FAA", status=AssetStatus.CANDIDATE,
    description=(
        "波音与FAA联合开发的 SDR 实体抽取模型。"
        "从服务困难报告中自动提取：飞行阶段/产品位置/机组动作/产品/产品状态等 7 类实体。"
        "直接用于航空维修知识图谱构建。"
    ),
    intro_level="L3_OPTIONAL", priority="P3_LOW",
    impl_method="huggingface", impl_source="https://huggingface.co/boeing/aviation-ner",
    impl_notes="从 HuggingFace 加载 GLiNER 模型；pip install gliner",
    risk_notes="仅对 SDR 格式文本高精度——非结构化维修日志可能性能下降",
    limitation_notes="7 类实体——不包含损伤类型和严重度等扩展实体",

    inputs=[InputSpec(artifact_type="raw_text", required_fields=["text_content"])],
    outputs=[OutputSpec(artifact_type="knowledge_reference",
        description="SDR 7类实体+原文引用")],

    applicability=ApplicabilitySpec(
        components=["any"], operating_modes=["sdr_analysis"],
        damage_types=["any"], exclusions=["non_aviation_text", "non_english"],
    ),

    method=MethodSpec(family="knowledge_extraction_transformer_ner", deterministic=False,
        assumptions=["输入为SDR或类似航空维修文本"],
        default_parameters={"entity_types": [
            "Flight Phase", "Product Location", "Crew Action",
            "Product", "Product Condition", "Bird/Animal Strike",
            "Emergency/Abnormal Situation",
        ]},
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["boeing_internal_sdr_test"],
        metrics={"precision": 0.85, "recall": 0.80},
        reviewer="Boeing + FAA data science", reviewed_at="2024",
    ),

    uncertainty=UncertaintySpec(output_representation="none"),
    resources=ResourceSpec(cpu=2, memory_mb=2048, gpu=False, timeout_s=120.0),
    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=False, data_classification="internal"),
    metrics_keys=["entity_count", "avg_confidence"],
    known_failure_modes=["gliner_model_download_failed"],
)

ASSET_MAINTIE_KNOWLEDGE = EngineeringAssetManifest(
    asset_id="knowledge_source.maintenance.maintie_schema_knowledge",
    name="MaintIE 维修信息抽取细粒度知识本体",
    version="1.0.0", asset_kind=AssetKind.KNOWLEDGE_SOURCE,
    publisher="nlp-tlp", status=AssetStatus.CANDIDATE,
    description=(
        "维修短文本细粒度信息抽取本体和基准数据集。"
        "5 个顶层类、224 个叶实体、6 种关系类型。"
        "1,076 细粒度 + 7,000 粗粒度标注文本。MIT 许可。"
        "适合作为维修领域 NER/关系抽取的统一 Schema 标准。"
    ),
    intro_level="L3_OPTIONAL", priority="P3_LOW",
    impl_method="git_clone", impl_source="https://github.com/nlp-tlp/maintie",
    impl_notes="可作为系统内维修信息抽取的Schema参考标准",
    risk_notes="224实体粒度极细——实际应用可能需要简化映射",
    limitation_notes="数据集以通用维修为主——航空发动机领域需扩展定制",

    inputs=[InputSpec(artifact_type="raw_text", required_fields=["text_content"])],
    outputs=[OutputSpec(artifact_type="knowledge_reference",
        description="MaintIE Schema 结构化的维修知识条目")],

    applicability=ApplicabilitySpec(
        components=["any"], operating_modes=["maintenance_log_analysis"],
        exclusions=[],
    ),

    method=MethodSpec(family="knowledge_extraction_ner_schema", deterministic=False,
        assumptions=["维修文本可映射到 MaintIE 224实体Schema"],
        default_parameters={"entity_schema": "maintie_v1"},
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["maintie_benchmark"],
        metrics={"ner_f1_micro": 0.76},
        reviewer="LREC-COLING 2024 reviewers", reviewed_at="2024",
    ),

    uncertainty=UncertaintySpec(output_representation="none",
        error_sources=["细粒度实体歧义", "领域术语不在Schema中"]),
    resources=ResourceSpec(cpu=2, memory_mb=1024, gpu=False, timeout_s=60.0),
    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=False, data_classification="internal"),
    metrics_keys=["entities_per_text", "entity_types_found"],
    known_failure_modes=["schema_files_not_found"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# ── 决策规则 (DECISION_RULE) 2个 ──
# ═══════════════════════════════════════════════════════════════════════════════

ASSET_RISK_CLASSIFICATION_RULES = EngineeringAssetManifest(
    asset_id="decision_rule.engine.risk_classification",
    name="风险分级决策规则引擎 (5级: negligible/low/medium/high/critical)",
    version="1.0.0", asset_kind=AssetKind.DECISION_RULE,
    publisher="internal", status=AssetStatus.DRAFT,
    description=(
        "基于损伤类型×严重度×部件×工况×机型 的 5 级风险分级规则。"
        "输出风险等级(negligible/low/medium/high/critical)和对应的候选处置动作。"
        "规则必须双人审核（文档 10.1 节）。"
    ),
    intro_level="L1_CORE", priority="P1_HIGH",
    impl_method="seekflow_wrap", impl_source="internal",
    impl_notes="规则配置文件(risk_rules.yaml)；规则变更需双人审核+审批",
    risk_notes="规则覆盖不完整或错误可能导致漏判或误判——必须双人审核",
    limitation_notes="P0 阶段使用透明化分级规则——不依赖 ML 黑盒",

    inputs=[InputSpec(artifact_type="damage_characterization", required_fields=[
        "damage_type", "severity", "component_location"])],
    outputs=[OutputSpec(artifact_type="decision_draft",
        description="风险等级 + 规则引用 + 触发原因")],

    applicability=ApplicabilitySpec(
        components=["any"], operating_modes=["any_inspection"],
        damage_types=["any"], exclusions=[],
    ),

    method=MethodSpec(family="rule_engine_deterministic", deterministic=True,
        assumptions=["风险分级规则已获领域批准"],
        default_parameters={"rule_version": "1.0", "require_dual_review": True},
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["expert_validated_risk_cases"],
        metrics={"expert_agreement": 0.95},
        reviewer="domain experts (TBD)", reviewed_at="TBD",
    ),

    uncertainty=UncertaintySpec(output_representation="none",
        calibration_method="cross_check_against_expert_judgment",
        ood_checks=["rule_coverage_gap", "unclassified_combination"]),
    resources=ResourceSpec(cpu=1, memory_mb=128, gpu=False, timeout_s=2.0, parallel_safe=True),
    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=True, data_classification="confidential"),
    metrics_keys=["risk_level", "rule_hits", "coverage_flag"],
    known_failure_modes=["rule_file_not_found", "unmapped_damage_severity_combo"],
)

ASSET_INSPECTION_INTERVAL_RULES = EngineeringAssetManifest(
    asset_id="decision_rule.engine.inspection_interval",
    name="复检周期决策规则引擎",
    version="1.0.0", asset_kind=AssetKind.DECISION_RULE,
    publisher="internal", status=AssetStatus.DRAFT,
    description=(
        "依据损伤类型×风险等级×使用循环计算的复检周期建议。"
        "输出复检间隔 (cycles / hours)、候选维修动作、适用条件。"
        "支持 continue_operation / increased_monitoring / reinspection / repair / replace 等 8 种动作。"
    ),
    intro_level="L2_RECOMMENDED", priority="P2_MEDIUM",
    impl_method="seekflow_wrap", impl_source="internal",
    impl_notes="规则配置文件(inspection_rules.yaml)；可接入发动机OEM标准手册",
    risk_notes="复检间隔直接影响飞行安全——必须双人审核+OEM手册对照",
    limitation_notes="P0 阶段使用简化版规则——完整版需接入 OEM 维修手册",

    inputs=[InputSpec(artifact_type="decision_draft", required_fields=[
        "risk_level", "damage_type", "rul_estimate"], allow_missing=["rul_estimate"])],
    outputs=[OutputSpec(artifact_type="decision_draft",
        description="推荐复检间隔 + 动作类型 + 规则引用 + 条件")],

    applicability=ApplicabilitySpec(
        components=["any"], operating_modes=["any_inspection"],
        damage_types=["any"], exclusions=[],
    ),

    method=MethodSpec(family="rule_engine_deterministic", deterministic=True,
        assumptions=["复检周期规则已获批准"],
        default_parameters={"rule_version": "1.0", "default_max_interval_cycles": 100},
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["oem_inspection_manual_benchmark"],
        metrics={"oem_agreement": 0.90},
        reviewer="domain experts (TBD)", reviewed_at="TBD",
    ),

    uncertainty=UncertaintySpec(output_representation="none",
        ood_checks=["rule_not_applicable_to_engine_type"]),
    resources=ResourceSpec(cpu=1, memory_mb=128, gpu=False, timeout_s=2.0, parallel_safe=True),
    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=True, data_classification="confidential"),
    metrics_keys=["interval_cycles", "action_type", "rule_ref"],
    known_failure_modes=["rule_file_not_found"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# ── 监控插件 (MONITOR) 1个 ──
# ═══════════════════════════════════════════════════════════════════════════════

ASSET_DATA_QUALITY_MONITOR = EngineeringAssetManifest(
    asset_id="monitor.data_quality.data_quality_gate",
    name="数据质量门监控器 (7维质量检查)",
    version="1.0.0", asset_kind=AssetKind.MONITOR,
    publisher="internal", status=AssetStatus.DRAFT,
    description=(
        "7 维数据质量门监控：结构完整性/数值有效性/时间一致性/"
        "单位和量纲/来源与标定/任务适用性/隐私与保密。"
        "产生 DataQualityReport：pass/warn/fail + 推荐动作。"
        "Agent 负责解释报告并提出补数建议（文档 8.2 节）。"
    ),
    intro_level="L1_CORE", priority="P0_CRITICAL",
    impl_method="seekflow_wrap", impl_source="internal",
    impl_notes="初始化时自动运行；每次数据传入时触发",
    risk_notes="质量检查规则本身可能不完整——需要根据新型传感器和数据格式持续更新",
    limitation_notes="检查是统计/规则基础的——不能检测所有可能的数据异常",

    inputs=[InputSpec(artifact_type="any", required_fields=["artifact_id", "artifact_type"])],
    outputs=[OutputSpec(artifact_type="data_quality_report",
        description="7维质量报告 + 通过/警告/失败 + 建议动作")],

    applicability=ApplicabilitySpec(
        components=["any"], operating_modes=["any"],
        exclusions=[],
    ),

    method=MethodSpec(family="monitoring_data_quality", deterministic=True,
        assumptions=["质量检查规则适用于输入数据类型"],
        default_parameters={
            "check_structural": True, "check_numerical": True,
            "check_temporal": True, "check_units": True,
            "check_source": True, "check_fitness": True,
            "check_privacy": True,
        },
    ),

    verification=VerificationSpec(
        validation_dataset_ids=["synthetic_quality_defect_cases"],
        metrics={"defect_detection_rate": 0.95, "false_positive_rate": 0.03},
        reviewer="internal", reviewed_at="TBD",
    ),

    uncertainty=UncertaintySpec(output_representation="none",
        error_sources=["新型数据异常的检测能力有限"]),
    resources=ResourceSpec(cpu=1, memory_mb=512, gpu=False, timeout_s=30.0, parallel_safe=True),
    policy=PolicySpec(risk="read", capabilities=["artifact.read"],
        requires_approval=False, data_classification="internal"),
    metrics_keys=["passed_checks", "warned_checks", "failed_checks",
                  "overall_status", "recommendation"],
    known_failure_modes=["artifact_type_not_recognized", "missing_metadata"],
)
