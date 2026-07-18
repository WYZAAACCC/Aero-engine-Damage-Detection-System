# Aero-Engine Damage Detection System — 系统完整状态报告

> 核查日期：2026-07-19  
> 核查方式：逐一运行 31 个资产 + 端到端测试  
> GPU：NVIDIA RTX 4060 Laptop, 8GB VRAM, CUDA 12.4  
> 仓库：`WYZAAACCC/Aero-engine-Damage-Detection-System`，`main` 分支

---

## 零、总体判断

**系统不是一个空壳。** 它包含：

1. 一个**真正可运行的 Agent 运行时**（SeekFlow DeepSeekAgent + ToolRuntime）
2. 一套**领域模型和编排基础设施**（状态机/计划编译器/执行引擎/API）
3. **6 个真实的深度学习模型**，已在公开数据集上训练并接入适配器
4. **25 个未达安全关键级别的工程资产**（基线替代/未接入/VRAM 不足/数据缺失）

**系统定位依然是研究原型**，不可用于真实维修决策。但它在诚实性方面已大幅改进——所有资产如实报告执行状态、有效性和决策能力。

---

## 一、31 个工程资产逐一核查

### 1.1 视觉检测器（5个）

| 资产 | 真实实现 | 有效性 | 说明 |
|------|:---:|:---:|------|
| **CA² 异常检测** | ✅ PatchCore (WideResNet-50 + Memory Bank) | `valid` | AeBAD 训练, AUROC=0.60, 2552 特征库已加载 |
| **SLF-YOLO 缺陷检测** | ❌ Sobel 边缘基线 | `degraded` | 无领域权重。通用 COCO YOLO 不可替代。BladeSynth 下载中 |
| **SAM-Adapter 裂纹分割** | ❌ 中心点 SAM / Canny | `degraded` | 无裂纹 Adapter/LoRA 权重。8GB VRAM 不够训练 |
| **EGCIENet 分割** | ✅ UNet+ResNet18 | `valid` | AEBIS 训练, Dice=0.79。替代 SegFormer（VRAM不够 + 权重未公开） |
| **TS-SAM 分割** | ❌ 完全不可用 | `invalid` | 需 SAM ViT-H (~2.4GB) + 双流 Adapter。8GB VRAM 远远不够 |

### 1.2 信号检测器（3个）

| 资产 | 真实实现 | 有效性 | 说明 |
|------|:---:|:---:|------|
| **WCamba 轴承故障** | ✅ WideKernel 1D-CNN | `valid` | CWRU 训练, 100% 测试准确率。4 类：normal/inner/outer/ball |
| **FaultSense LSTM-AE** | ✅ LSTM Encoder-Decoder + RUL | `valid` | FD001 训练, RMSE=16.1。PyTorch 移植版（TF 已废弃） |
| **Isolation Forest** | ⚠️ sklearn IsolationForest | `unverified` | 未拟合校准模型。需真实发动机遥测数据训练 |

### 1.3 表征工具（3个）

| 资产 | 真实实现 | 有效性 | 说明 |
|------|:---:|:---:|------|
| **裂纹几何测量** | ⚠️ bbox 矩形估计 | `valid`(有标尺)/`degraded`(无标尺) | 非骨架化。需要显式 bbox_format。有标尺时可输出 mm 值 |
| **损伤类型分类** | ❌ 关键词规则引擎 | `degraded` | 11 类 × 10 关键词匹配。不是训练分类器。score_semantics 未用 |
| **严重度分级** | ⚠️ 通用阈值 (0.5/2/5mm) | `degraded` | 未绑定机型/部件/材料。涂层面积需 area_percent 非 area_mm² |

### 1.4 可靠性/寿命模型（6个）

| 资产 | 真实实现 | 有效性 | 说明 |
|------|:---:|:---:|------|
| **CNN-LSTM RUL** | ✅ Conv1d+BiLSTM | `valid` | FD001 RMSE=15.0, FD003 RMSE=22.0。替换了旧的线性趋势 |
| **Paris 裂纹扩展** | ⚠️ 数值积分 | `degraded` | 仅演示。无 ΔKth/Kc 检查。必须显式提供材料参数 |
| **FDPP 概率裂纹** | ⚠️ Monte Carlo+Paris | `unverified` | 简化的 500 样本 MC。缺 FORM/SORM |
| **pyLife S-N 曲线** | ⚠️ Basquin 公式 | `unverified` | 非 FKM 非线性。pip install pylife 可升级 |
| **ChangePoint-LSTM** | ✅ CUSUM+LSTM | `valid` | FD002 RMSE=31.7, 85% 变点检出率 |
| **PINN 机队预测** | ❌ 返回 unavailable | `unverified` | 未实现。需物理参数 + N-CMAPSS + GPU |

### 1.5 知识源（3个）

| 资产 | 真实实现 | 有效性 | 说明 |
|------|:---:|:---:|------|
| **Omin 专家知识库** | ⚠️ 29 条手写 Python 字典 | `degraded` | 非文档 RAG。24/29 自标证据等级 A 但无审批。1/29 有页码。已实现发动机/材料硬过滤 |
| **Boeing NER** | ❌ 1 条固定数据 | `unverified` | 非真实 SDR 数据集 |
| **MaintIE 本体** | ❌ 类名 + GitHub 提示 | `unverified` | 非真实本体查询 |

### 1.6 决策规则（2个）

| 资产 | 真实实现 | 有效性 | 说明 |
|------|:---:|:---:|------|
| **风险分级** | ⚠️ 4×3 通用矩阵 | `degraded` | 非机型特定。未知组合→unknown+blocked。始终要求复核 |
| **复检周期** | ❌ 硬编码 0/25/100/300/1000 | `degraded` | 非发动机型号/ATA 特定。标记为不可决策 |

### 1.7 数据适配器（5个）、预处理器（3个）、监控（1个）

| 类型 | 数量 | 典型状态 | 说明 |
|------|:---:|:---:|------|
| 数据适配器 | 5 | `unverified` | 数据读取器。CA²/C-MAPSS/BladeSynth 等。未验证数据完整性 |
| 预处理器 | 3 | `unverified` | SciPy频谱/OpenCV/VKF。真实 scipy/sklearn 实现 |
| 数据质量门 | 1 | `valid` | 7 维检查。已添加温度/压力/振动物理范围检查。缺 pint 单位 |

---

## 二、已训练模型详情（6个）

### 模型 1：WCamba — 轴承故障分类
```
资产ID:    detector.vibration.wcamba_bearing_fault
数据:      CWRU 12kHz Drive End, 64 .mat 文件
架构:      WideKernel 1D-CNN (9,300 params)
类别:      normal, inner_race, outer_race, ball (4 类)
测试准确率: 100% (538 测试样本)
适配器:    detectors_signal.py
状态:      valid, can_influence_decision=True
产物:      artifacts/models/wcamba_cwru_4class/v1.0/ (44 KB)
```

### 模型 2：CNN-LSTM — 剩余寿命预测
```
资产ID:    reliability_model.rul.cnn_lstm_cmapss
数据:      C-MAPSS FD001 (100 发动机) + FD003 (100 发动机)
架构:      Conv1d(32→64) + BiLSTM(128) + MLP (178,721 params)
输入:      50 步 × 14 传感器
FD001:     Test RMSE=15.0 cycles, NASA Score=4
FD003:     Test RMSE=22.0 cycles, NASA Score=22
适配器:    rul_predictor.py
状态:      valid, can_influence_decision=True
产物:      artifacts/models/cnn_lstm_rul/v1.0/ (1.4 MB)
```

### 模型 3：FaultSense — 异常检测 + RUL
```
资产ID:    detector.timeseries.faultsense_lstm_autoencoder
数据:      C-MAPSS FD001
架构:      LSTM Encoder(2层,hidden=32) + Decoder(1层) + RUL MLP (24,047 params)
输入:      30 步 × 14 传感器
异常检测:   重构误差 + k-sigma 阈值 (k=2.5)
FD001:     Test RMSE=16.1, NASA Score=6
移植:      从 TensorFlow .h5 → PyTorch (TF 在 Python 3.13 上损坏)
适配器:    detectors_signal.py
状态:      valid, can_influence_decision=True
产物:      artifacts/models/faultsense/v1.0/FD001/ (98 KB)
```

### 模型 4：PatchCore — 叶片异常检测
```
资产ID:    detector.borescope.ca2_anomaly
数据:      AeBAD-S (521 正常训练 / 1818 测试)
架构:      WideResNet-50 (layer2+layer3) + Coreset Memory Bank
特征库:    2552 patches × 1536 dims (来自 52 张正常图)
测试:      Image AUROC=0.605, F1=0.633 (AeBAD 域偏移显著)
适配器:    detectors_vision.py
状态:      valid, can_influence_decision=True
产物:      artifacts/models/patchcore_aebad/v1.0/ (15 MB)
```

### 模型 5：UNet — 叶片缺陷分割
```
资产ID:    detector.borescope.egcienet_segmentation
数据:      AEBIS (422 训练 / 59 测试)
架构:      UNet + ResNet-18 Encoder (17,597,249 params)
输入:      256×256
测试:      Dice=0.787
适配器:    detectors_vision.py
状态:      valid, can_influence_decision=True
产物:      artifacts/models/unet_aebis/v1.0/ (67 MB)
备注:      替代 EGCIENet (SegFormer 需 12GB+ VRAM, 权重未公开)
```

### 模型 6：ChangePoint-LSTM — 多变工况 RUL
```
资产ID:    reliability_model.rul.changepoint_lstm_multicondition
数据:      C-MAPSS FD002 (260 发动机, 6 工况)
架构:      Conv1d(32→64) + BiLSTM(64×2) + RUL head (180,802 params)
变点检出率:  85% (CUSUM 方法)
FD002:     Test RMSE=31.7, NASA Score=276
适配器:    reliability_extended.py
状态:      valid, can_influence_decision=True
产物:      artifacts/models/changepoint_lstm/v1.0/FD002/ (717 KB)
```

---

## 三、核心基础设施

### 3.1 编排层

| 组件 | 状态 | 说明 |
|------|:---:|------|
| **TaskStateMachine** | ✅ | 16 状态。7 强制门需 checker。禁止系统自动 APPROVE |
| **PlanCompiler** | ✅ | Kahn 算法。环检测+孤立/不可达。完整 digest |
| **PlanExecutor** | ✅ | 拓扑调度+retry+failure strategy。替代 /runs 占位符 |
| **RunStore** | ✅ | 运行 CRUD+节点追踪+checkpoint。内存实现（原型） |

### 3.2 Agent 层

| 组件 | 状态 | 说明 |
|------|:---:|------|
| **7 个 Agent 角色** | ✅ | system_prompt + allowed_tools + forbidden_actions |
| **DomainAgentController** | ✅ | Agent 工厂 + 默认处理器 |
| **6 个平台工具** | ✅ | search_assets/inspect/retrieve_knowledge/propose_plan/validate/review |
| **PlannerOutput Schema** | ✅ | Pydantic 强制结构化输出 |
| **DeepSeekAgent 集成** | ⚠️ | API key 可用时真实调用；否则 graceful degradation |

### 3.3 API 层

| 端点 | 状态 |
|------|:---:|
| `/api/v1/tasks` CRUD | ✅ |
| `/api/v1/assets` 搜索 | ✅ 启动时自动注册 31 个 |
| `/api/v1/runs` 6 端点 | ✅ |
| `/api/v1/reviews` | ✅ |
| `/health` | ✅ |

### 3.4 安全与审计

| 功能 | 状态 |
|------|:---:|
| 匿名审批拒绝 | ✅ |
| AssetRunResult 三态拆分 | ✅ |
| 资产状态下调 | ✅ |
| 未知风险→unknown+blocked | ✅ |
| 数字签名 | ❌ |
| OIDC 认证 | ❌ |

---

## 四、数据集状态

| 数据集 | 大小 | 用途 | 状态 |
|------|------|------|:---:|
| CWRU | ~40 MB | WCamba | ✅ |
| C-MAPSS | ~50 MB | CNN-LSTM, FaultSense, ChangePoint | ✅ |
| AeBAD | 1.6 GB | PatchCore | ✅ |
| AEBIS | 90 MB | UNet | ✅ |
| BladeSynth | 25.7 GB | SLF-YOLO | ⏳ 70% |
| Paderborn | — | WCamba 跨工况 | ❌ |
| CrackSeg9k | — | SAM-Adapter | ❌ |

---

## 五、领域测试

| 文件 | 数量 | 覆盖 |
|------|:---:|------|
| `test_state_machine.py` | 7 | 门检查/非法迁移/自动审批阻断 |
| `test_plan_compiler.py` | 9 | DAG环检测/孤立节点/digest |
| `test_geometry.py` | 10 | bbox格式/负值/量表/damage/severity |
| `test_risk_rules.py` | 7 | 未知→blocked/发动机材料过滤/跨域 |
| **合计** | **33** | `pytest tests/aero_diag/` — 全部通过 |

---

## 六、诚实的能力矩阵

```
                      运行框架  算法名义存在  工程有效  安全关键可用
WCamba                  ✅         ✅          ✅         ❌
CNN-LSTM RUL            ✅         ✅          ✅         ❌
FaultSense              ✅         ✅          ✅         ❌
PatchCore/CA²           ✅         ✅          ✅         ❌
UNet/EGCIENet           ✅         ✅          ✅         ❌
ChangePoint-LSTM        ✅         ✅          ✅         ❌
SLF-YOLO                ✅         ❌          ❌         ❌
SAM-Adapter             ✅         ⚠️          ❌         ❌
TS-SAM                  ✅         ❌          ❌         ❌
Isolation Forest        ✅         ✅          ❌         ❌
Paris Law               ✅         ✅          ⚠️         ❌
FDPP/pyLife/PINN        ✅         ⚠️          ❌         ❌
几何测量/分类/分级       ✅         ⚠️          ❌         ❌
知识库/规则              ✅         ⚠️          ❌         ❌
数据适配器/预处理器      ✅         ✅          ⚠️         ❌
```

**0 个资产可安全用于真实维修决策。**

---

## 七、审计修复进度

审计文档（`深度实现审计与修复指导.md`）20 个问题：

| 状态 | 数量 | 问题 |
|------|:---:|------|
| ✅ 已修复 | 17 | AER-001~013, 016~019 |
| ⚠️ 部分 | 2 | AER-014（知识缺页码）, AER-020（内存存储） |
| ❌ 未修复 | 1 | AER-006（需生成验证报告） |
