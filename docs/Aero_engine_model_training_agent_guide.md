# 航空发动机损伤检测系统：模型训练数据与自动化实施指导书

**目标仓库：** `WYZAAACCC/Aero-engine-Damage-Detection-System`  
**证据核验日期：** 2026-07-18  
**文档用途：** 直接交付代码 Agent，用于下载数据、固化数据版本、转换标注、编写训练脚本、执行训练、评估、导出权重并接入现有系统。  
**适用范围：** 研究原型和工程验证；不构成航空维修放行、适航认证或真实机队寿命决策依据。

---

## 0. 必须先读：这份文档能保证什么、不能保证什么

这份指导书通过以下方式，尽量把“训练不起来、权重接不上、指标虚高、数据泄漏、类别错位”等常见失败提前消除：

1. 按目标仓库的**真实代码接口**逐一对齐模型、输入、权重路径、类别和窗口长度；
2. 优先选用官方或论文作者发布的数据与代码，并给出来源、格式、标注状态、许可证和下载策略；
3. 对公开数据与航空发动机真实目标域之间的差距作明确分级；
4. 为每个模型规定数据转换、划分、训练、评估、导出和接入验收条件；
5. 要求所有下载、预处理和训练产物可追溯、可复现、可重新执行；
6. 对没有充分公开数据支撑的模型明确标注为“研究级”或“必须使用自有数据”。

但任何人都不能诚实地保证：外部链接永远可访问、第三方数据许可永远不变、特定 GPU 环境一定兼容、公开数据训练出的模型一定能在真实发动机上达到生产要求。因此代码 Agent 必须实施本文定义的**失败即停止（fail-fast）门禁**，不得遇到缺数据、缺标注、格式不明或指标不达标时伪造成功权重。

最终目标分为三层：

- **L1 软件可用：** 模型能加载、能推理、输入输出契约正确、不会静默退化；
- **L2 数据集内有效：** 在独立测试集上达到本文最低门槛，并且不存在明显泄漏；
- **L3 目标域有效：** 在真实叶片、真实发动机或真实机队数据上独立验证。公开数据通常只能帮助达到 L1-L2，L3 必须依赖目标单位数据和专家验证。

---

# 1. 执行总纲

## 1.1 训练顺序

代码 Agent 必须按下列顺序执行，不得同时铺开全部模型：

### 阶段 0：修复代码契约

先修复以下问题，否则训练完成也无法正确接入：

1. 建立统一模型注册表，取消散落的硬编码权重路径；
2. `WCambaBearingFaultDetector` 的输入长度从目标代码的 2048 对齐到上游网络的 1024，或在配置中显式版本化；
3. WCamba 类别必须加入 `normal`，第一版不得保留没有数据支撑的 `cage` 类；
4. FaultSense 从错误的 TensorFlow `.h5` 加载方式改为 PyTorch `.pt`；
5. Isolation Forest 必须加载已拟合的 `Pipeline/Scaler/Model/Threshold`，未拟合时返回 `unavailable`，不得伪装为成功；
6. CNN-LSTM RUL 必须替换当前线性外推占位实现；
7. SAM-Adapter 与 TS-SAM 必须真正构造各自网络，不得把“基础 SAM + 中心点提示”标记成相应模型；
8. 每个适配器必须区分 `success`、`degraded`、`unavailable`，且只有真实训练模型允许 `success`；
9. 每个模型输出必须附带 `model_id`、`model_version`、`weight_sha256`、`dataset_manifest_sha256`、`preprocessing_version`。

### 阶段 1：优先完成的六条真实闭环

1. **叶片异常检测：** AeBAD + MMR/PatchCore 后端，接入 `CA2AnomalyDetector`；
2. **叶片缺陷分割：** AEBIS + EGCIENet；
3. **叶片缺陷检测：** BladeSynth + AEBIS，训练 SLF-YOLO；
4. **轴承诊断：** CWRU 四类版 WCamba；
5. **发动机退化/RUL：** C-MAPSS 上重训 FaultSense；
6. **RUL 回归：** C-MAPSS 上训练真正 CNN-LSTM。

### 阶段 2：扩展模型

1. SAM-Adapter 裂纹分割；
2. TS-SAM 叶片缺陷分割；
3. Paderborn 跨工况轴承模型；
4. ChangePoint-LSTM；
5. 基于自有正常遥测的 Isolation Forest。

### 阶段 3：仅在具备目标域数据后实施

1. N-CMAPSS 大规模复杂工况训练；
2. PINN 舰队寿命模型；
3. Paris 定律、S-N、裂纹扩展等物理模型的型号级校准；
4. 航空维修决策阈值与严重度分级。

---

## 1.2 统一目录结构

代码 Agent 必须在仓库根目录创建：

```text
training/
├── README.md
├── environments/
│   ├── vision-modern.lock
│   ├── ts-sam-legacy.lock
│   ├── signal-modern.lock
│   └── rul-modern.lock
├── configs/
│   ├── datasets/
│   ├── models/
│   └── experiments/
├── scripts/
│   ├── download/
│   ├── audit/
│   ├── convert/
│   ├── split/
│   ├── train/
│   ├── evaluate/
│   ├── export/
│   └── integrate/
├── src/aero_training/
│   ├── datasets/
│   ├── models/
│   ├── metrics/
│   ├── registry/
│   └── common/
├── tests/
│   ├── data/
│   ├── models/
│   ├── integration/
│   └── golden/
└── reports/
```

仓库外或被 `.gitignore` 排除的工作区：

```text
artifacts/
├── raw/                 # 原始下载，只读
├── extracted/           # 原始解压
├── canonical/           # 统一格式
├── splits/              # 仅保存 manifest，不复制大文件
├── caches/
├── runs/
├── models/
└── reports/
```

原始数据禁止直接修改。任何修复、重命名、缩放、掩膜转换都必须生成到 `canonical/`，并写入转换日志。

---

## 1.3 数据来源和许可证门禁

每个数据集下载后必须生成：

```text
dataset_root/
├── SOURCE.md
├── LICENSE_SNAPSHOT.txt
├── download_manifest.json
├── sha256sums.txt
├── raw/
└── canonical/
```

`download_manifest.json` 至少包含：

```json
{
  "dataset_id": "bladesynth_v1",
  "official_landing_page": "...",
  "downloaded_at_utc": "...",
  "source_type": "official|author_repo|institutional|mirror",
  "source_version": "...",
  "license": "...",
  "license_verified": true,
  "files": [
    {"path": "...", "bytes": 0, "sha256": "..."}
  ]
}
```

规则：

- 官方源优先；
- 镜像只能在官方源不可用时使用，并在 `SOURCE.md` 说明；
- 许可证未找到或不允许目标用途时，数据可用于本地研究但不得进入可分发权重；
- 禁止代码 Agent 自动点击、绕过登录、破解网盘或接受未知协议；
- 对 Google Drive、百度网盘、天翼云等需要人工授权的来源，脚本应输出清晰的人工步骤并停止，而不是下载一个 HTML 错误页后继续训练；
- 下载后必须验证文件类型、压缩包完整性、样本数量、图像可读率、标注配对率和重复率。

---

## 1.4 数据划分总规则

所有模型必须遵守：

1. **先分组，后切窗/裁块。** 同一叶片、同一发动机、同一视频、同一原始图、同一轴承或同一运行轨迹不得跨训练/验证/测试；
2. 相邻视频帧不得随机分散到不同集合；
3. 轴承振动的重叠窗口不得随机切分；
4. C-MAPSS 必须按 `unit_id` 分组；
5. 合成数据必须按渲染种子、CAD 实例或场景模板分组，避免同一场景的微小变体跨集合；
6. 所有阈值只允许在验证集上确定；测试集只运行一次最终评估；
7. 目标域真实测试集不得用于早停、调参、阈值选择或类别合并决策；
8. 数据集官方自带测试集时必须保留，同时另设训练内部验证集；
9. 生成 `split_audit.json`，自动检查 group_id 交集为零。

---

## 1.5 通用样本清单格式

视觉任务统一使用 `manifest.parquet`：

```text
sample_id              string
source_dataset         string
source_version         string
source_path             string
image_path              string
mask_path               nullable string
annotation_path         nullable string
split                    train|val|test|external_test
group_id                 string
engine_id                nullable string
blade_id                 nullable string
video_id                 nullable string
frame_id                 nullable string
domain                   real|replica|synthetic|generic_metal|generic_crack
is_synthetic             bool
width                     int
height                    int
labels                    JSON string
annotation_type           none|image_label|bbox|polygon|binary_mask|multiclass_mask
annotation_quality        original|converted|weak|manual_verified
license_id                string
sha256_image              string
sha256_annotation         nullable string
```

时序任务统一使用：

```text
record_id, asset_id, unit_id, bearing_id, cycle, timestamp,
operating_condition_id, sample_rate_hz, channel_schema_version,
source_file, split, group_id, label, rul, sha256_source
```

---

## 1.6 模型产物统一格式

每次可接入的训练必须输出一个不可变目录：

```text
artifacts/models/<model_id>/<semantic_version>/
├── model.pt | model.pth | best.pt | model.onnx | model.joblib
├── config.yaml
├── preprocessing.json
├── class_map.json
├── thresholds.json
├── metrics.json
├── data_provenance.json
├── environment.lock
├── model_card.md
├── weight.sha256
├── golden_inputs/
├── golden_outputs/
└── integration_report.json
```

`model_card.md` 必须说明：任务、数据、划分、模型结构、输入、输出、限制、已知失败模式、不能用于什么、指标和目标域差距。

---

# 2. 数据源总表

| ID | 数据集 | 任务 | 真实/合成 | 规模与格式 | 是否有标注 | 推荐等级 |
|---|---|---|---|---|---|---|
| D01 | AeBAD-S / AeBAD-V | 叶片无监督异常检测 | 真实叶片 + 3D 打印复刻件 | 图像目录；单叶片与视频子集 | 训练正常标签；测试异常标签；异常掩膜目录 | 主数据，异常检测首选 |
| D02 | CA² Blade Dataset | 多发动机孔探图像异常检测 | 真实域，官方说明为多发动机 | 官方网盘；公开 README 对结构描述有限 | 需下载后审计 | 外部测试/补充正常库 |
| D03 | BladeSynth | 检测、分割、异常检测 | 合成 | 12,500 张，1024×1024；5 类，每类 2,500 | 缺陷图配像素掩膜；Normal 无缺陷掩膜 | 高价值辅助，不得单独作为最终验证 |
| D04 | AEBIS | 叶片表面缺陷分割/分类 | 真实服役叶片 | `AEBIS.zip`、`AEBIS_Class.zip`；Labelme JSON | 有现成多边形/类别标注，但需转换与审计 | 真实叶片分割主数据 |
| D05 | NEU-DET | 钢表面缺陷检测 | 通用金属 | 1,800 张灰度 200×200，6 类 | XML 边界框 | 仅预训练/管线测试 |
| D06 | GC10-DET | 金属表面缺陷检测 | 通用金属 | 公开版本数量存在差异；灰度大图；10 类 | XML 边界框，部分来源称有像素标注 | 可选预训练；必须版本审计 |
| D07 | CrackSeg9k | 裂纹分割 | 道路/墙体等非叶片 | 约 9.1k-9.3k，V4 分包 | 二值掩膜 | 裂纹形态预训练 |
| D08 | DeepCrack | 裂纹分割 | 路面/混凝土等非叶片 | 537 RGB；300 train/237 test | 手工二值掩膜 | 裂纹预训练和回归测试 |
| D09 | CFD | 裂纹分割 | 道路 | 118 张，约 480×320 | 手工轮廓/掩膜 | 小型外部泛化测试 |
| D10 | CWRU Bearing | 轴承故障分类 | 实验台 | MATLAB `.mat`；多采样率、多负载 | 正常、内圈、外圈、滚动体；无保持架类 | WCamba 第一主数据 |
| D11 | Paderborn Bearing | 轴承故障分类 | 实验台，含人工与真实损伤 | `.mat`/压缩包；振动与电流同步 | 健康、内圈、外圈等；由官方元数据定义 | WCamba 跨域第二数据 |
| D12 | XJTU-SY | 轴承退化/RUL | 加速寿命实验 | 15 个轴承；CSV 两振动通道；25.6 kHz，每分钟 32,768 点 | 故障部位与寿命信息；不是平衡窗口分类标签 | 轴承退化/异常/RUL，不作第一版四类分类主数据 |
| D13 | NASA C-MAPSS | 涡扇发动机 RUL/异常 | 仿真 | FD001-FD004；空格文本，26 列 | 训练轨迹到失效；测试 RUL 文件 | FaultSense、CNN-LSTM、ChangePoint 主数据 |
| D14 | NASA N-CMAPSS | 大规模涡扇退化/RUL | 高保真仿真 + 真实飞行工况 | HDF5；8 个子集，大规模时序 | 健康/故障/退化相关变量 | 二阶段复杂工况与 PINN 研究 |
| D15 | UCI Gas Turbine CO/NOx | 燃气轮机运行状态 | 工业地面燃机 | 36,733 条，11 个小时聚合传感量 | 排放回归目标，无故障标签 | 仅运行工况/异常算法辅助，不能替代航空发动机故障数据 |
| D16 | 自有孔探数据 | 检测/分割/目标域验证 | 真实目标域 | 原图、视频、部件元数据 | 必须由专家标注 | 达到 L3 的必要数据 |
| D17 | 自有发动机遥测 | IF、RUL、异常 | 真实目标域 | 原始传感器与事件记录 | 正常期、维修事件、故障确认 | 生产 IF/RUL 的必要数据 |
| D18 | 材料与载荷试验数据 | 物理寿命模型 | 真实材料/部件 | Paris/S-N/载荷谱/几何/温度 | 工程试验参数 | 物理模型校准必需 |

---

# 3. 视觉模型一：CA²/叶片异常检测资产

## 3.1 当前代码与训练目标

目标仓库 `CA2AnomalyDetector` 当前实际上是 ImageNet ResNet50 特征提取器，缺少正常样本特征库。它不能只靠一个分类权重完成异常检测。正确闭环至少包括：

- 正常训练图像；
- 图像对齐或多尺度局部特征；
- 正常特征 memory bank；
- KNN/重构异常分数；
- 图像级与像素级阈值；
- 正常域偏移验证。

因此推荐保留资产 ID，但把后端改造成可配置：

```yaml
asset_id: detector.borescope.ca2_anomaly
backend: mmr | patchcore
input_size: 224_or_backend_specific
reference_bank_required: true
```

## 3.2 主数据：AeBAD

**位置：** 官方 MMR 仓库 README 的数据下载入口。  
**官方代码：** `https://github.com/zhangzilongc/MMR`  
**任务匹配：** 极高。该数据专门研究航空发动机叶片异常检测及正常域偏移。  
**子集：** AeBAD-S（单叶片）和 AeBAD-V（视频）。  
**已知结构示例：**

```text
AeBAD/
├── AeBAD_S/
│   ├── train/good/<domain_or_background>/...
│   ├── test/<anomaly_type_or_good>/...
│   └── ground_truth/<anomaly_type>/...
└── AeBAD_V/
    ├── train/good/<video_train>/...
    └── test/<video>/<normal_or_anomaly>/...
```

AeBAD 的关键难点是训练与测试正常样本之间存在光照、视角和背景域偏移；样本未严格对齐且尺度不同。代码 Agent 不得使用只适合对齐工业图像的简单全局均值向量作为最终方案。

### 下载与审计任务

创建：

```text
training/scripts/download/download_aebad.py
training/scripts/audit/audit_aebad.py
training/scripts/convert/prepare_aebad.py
```

下载脚本必须：

1. 从官方 MMR README 解析当前下载链接；
2. 支持 `gdown` 或人工下载后导入；
3. 识别网盘错误页；
4. 记录压缩包 SHA256；
5. 解压后统计 AeBAD-S/AeBAD-V 每个目录样本数；
6. 验证所有掩膜与图像尺寸一致；
7. 用感知哈希检查近重复；
8. 从路径构造 `domain_id`、`video_id`、`anomaly_type`；
9. 不改变官方测试划分。

### 推荐训练方案

**方案 A：复现 MMR，优先。**

- 克隆官方 MMR 代码并固定 commit；
- 使用官方 AeBAD-S/AeBAD-V 配置；
- 下载 README 指定的 MAE ViT 预训练骨干；
- 首先运行官方评估脚本确认环境和数据目录正确；
- 再把 MMR 模型封装进本系统；
- 导出图像级异常分数、像素异常图和阈值。

**方案 B：PatchCore，工程备选。**

- 使用 ImageNet 预训练 WideResNet50/ResNet18 局部 patch 特征；
- 只用训练正常样本建立 memory bank；
- 使用 coreset subsampling；
- 验证集确定图像阈值和像素阈值；
- 单独报告每个域、每种异常和总体指标。

### 划分

- 官方 test 保持不变；
- 从官方 train/good 中按背景/拍摄序列分组划出 10%-20% 正常验证集；
- AeBAD-V 按整段视频分组；
- 禁止逐帧随机切分；
- BladeSynth 正常图只能作为额外训练域，不能混入 AeBAD 测试。

### 指标

至少：

- Image AUROC；
- Pixel AUROC；
- PRO/AUPRO；
- FPR@95TPR；
- 正常域误报率；
- 每个异常类别召回；
- 每个背景/视角域指标。

### 接入产物

```text
models/ca2_anomaly_mmr/<version>/
├── model.pth
├── backbone.pth
├── memory_bank.pt          # 若后端需要
├── thresholds.json
├── preprocessing.json
└── model_card.md
```

### 验收

- 在完全没有异常训练图的情况下可完成训练；
- 测试 good 的误报率有明确报告；
- 不得把 ImageNet backbone 当成训练完成权重；
- 缺少 memory bank 或阈值时加载器必须失败；
- 与当前亮度/方差基线相比，必须在 AeBAD 官方测试上显著更好；
- 最终外部测试至少加入 CA² 或自有真实孔探图像。

## 3.3 补充数据：CA² Blade Dataset

**位置：** `https://github.com/changniu54/CA2`  
官方 README 提供多发动机叶片孔探图像的数据下载入口，但公开页面对目录、精确数量、异常类型和许可证说明不充分，且官方代码发布状态有限。

代码 Agent 必须把它当作“下载后审计的数据”，不得在下载前假设其样本数量和标注形式。执行：

1. 保存官方 README 和下载页面快照；
2. 统计目录树和文件类型；
3. 打开随机 100 张图像人工缩略图审计；
4. 判断是否只有图像级标签、是否有像素掩膜；
5. 判断是否能识别发动机/叶片/视频 group；
6. 许可证不明时仅用于本地研究与外部测试；
7. 不能把 CA² 数据自动合并进 AeBAD 训练，先做独立域评估。

## 3.4 合成辅助：BladeSynth Normal

BladeSynth 的 Normal 类可以扩展正常表面变化，但不应替代真实正常叶片。建议最大占正常训练池的 30%，并使用 domain-balanced sampler，防止模型学会“合成渲染风格”。

---

# 4. 视觉模型二：SLF-YOLO 叶片缺陷检测

## 4.1 当前代码要求

目标适配器要求 `SLF-YOLO/weights/best.pt`，无权重时用 Sobel 基线。训练必须输出 Ultralytics 可加载的 `.pt`，并且 `class_map.json` 与适配器完全一致。

## 4.2 推荐类别本体

第一版统一检测类别：

```json
{
  "0": "dent",
  "1": "nick",
  "2": "scratch",
  "3": "corrosion",
  "4": "crack",
  "5": "pit",
  "6": "fracture",
  "7": "edge_deformation",
  "8": "other_damage"
}
```

规则：

- `normal` 不是目标框类别；
- `scratch` 不能自动映射为 `crack`；
- `nick`、`dent` 不得因视觉相似自动合并；
- AEBIS 原始类名需通过 `label_mapping.yaml` 显式映射；
- 无法可靠映射的标注进入 `other_damage` 或被排除，并记录原因；
- 每个映射至少人工审计 50 个实例。

## 4.3 主数据 A：BladeSynth

**位置：** `https://doi.org/10.6084/m9.figshare.28658603`  
**规模：** 25.74 GB；12,500 张 1024×1024；Normal、Dent、Nick、Scratch、Corrosion 各 2,500。  
**标注：** 缺陷图有自动生成且校验的像素掩膜。  
**许可证：** CC BY。  
**用途：** 检测/实例分割预训练和数据扩充。

### 下载

通过 Figshare API 发现实际文件而非硬编码临时 URL：

```python
GET https://api.figshare.com/v2/articles/28658603
```

记录每个文件的官方大小、下载 URL 和官方校验信息；支持断点续传。下载前检查至少 35 GB 可用空间，预处理和缓存建议准备 80 GB。

### 转为 YOLO 检测标签

对每个非空二值掩膜：

1. 连通域分析；
2. 删除小于图像面积 `1e-5` 的噪声区域；
3. 每个连通域生成 bbox；
4. 过度碎片化的腐蚀掩膜可合并为最小包围框，同时保留原分割标签；
5. 输出 YOLO normalized bbox：

```text
<class_id> <xc/W> <yc/H> <w/W> <h/H>
```

同时保留 YOLO segmentation polygon，便于后续实例分割。

### 合成数据划分

不要简单随机 60/40。优先读取生成元数据；如没有 scene seed，则以文件名、背景、相机位姿相似性和图像感知哈希聚类生成 `scene_group_id`。按 group 切分 70/15/15。最终真实测试不使用 BladeSynth。

## 4.4 主数据 B：AEBIS

**位置：** `https://github.com/Newbiejy/EGCIENet_In-service-blade-defect-detection/tree/main/Dataset`  
**文件：** `AEBIS.zip` 与 `AEBIS_Class.zip`。  
**标注：** Labelme JSON，多边形可转换为 mask 和 bbox。  
**注意：** `AEBIS.zip` 是论文相关版本；`AEBIS_Class.zip` 是早期分类目录，类别不平衡且部分类型被合并。两者不可静默混用。

转换：

```bash
python training/scripts/convert/convert_labelme_to_canonical.py \
  --input artifacts/extracted/aebis \
  --output artifacts/canonical/aebis \
  --mapping training/configs/datasets/aebis_label_mapping.yaml
```

转换器必须：

- 支持 polygon、rectangle 等 Labelme shape；
- 检测自交多边形；
- 将同一对象的多段标注合并；
- 保留原始 label；
- 生成 bbox、binary mask、可选 multiclass mask；
- 对无标注图像明确区分“正常”与“漏标”；
- 输出可视化叠加图供人工审核。

## 4.5 辅助预训练：NEU-DET 与 GC10-DET

### NEU-DET

- 1,800 张灰度 200×200；
- 6 类，每类 300；
- XML bbox；
- 是热轧钢表面，不是发动机叶片。

仅用于检测骨干预训练和数据转换测试。不得把其六类直接映射为叶片缺陷。预训练后替换检测头，在 BladeSynth/AEBIS 上重新训练。

### GC10-DET

- 10 类金属表面缺陷；
- 原始论文报告 3,570 张大尺寸灰度图，但公开流通版本有 2,300 与 3,570 等差异；
- 常见标注为 XML bbox，部分项目描述有像素级标注。

代码 Agent 必须把版本差异当成风险：记录实际 commit、实际文件数、实际标注数，不能在报告中混用论文规模和下载规模。该数据仅作通用金属预训练。

## 4.6 训练策略

分两阶段：

### 预训练阶段

```text
可选 NEU-DET + GC10-DET -> 通用金属缺陷特征
```

### 领域训练阶段

```text
BladeSynth + AEBIS -> 统一叶片类别
```

建议：

- 先复现 SLF-YOLO 上游训练入口；
- 使用 COCO 预训练或通用金属预训练权重初始化；
- 640 或 1024 输入，根据显存和小缺陷尺寸选择；
- 保留高分辨率随机裁块策略；
- 使用 source-balanced sampler，防止 12,500 张合成图淹没 AEBIS；
- 数据增强应模拟孔探场景：亮度、色温、轻微模糊、镜面高光、暗角、JPEG、有限旋转；
- 禁止会改变缺陷物理形态的强透视和过度形变；
- 训练时记录每源、每类 AP。

## 4.7 验收

- AEBIS 真实测试上的 mAP50、mAP50-95、precision、recall；
- 小目标 AP；
- 每类混淆；
- 合成训练/真实测试差距；
- 不得只报告 BladeSynth 测试成绩；
- 至少提供 100 张真实图误检审查；
- 导出的 `best.pt` 用目标适配器运行 20 个 golden samples，输出类别与坐标必须一致。

---

# 5. 视觉模型三：EGCIENet 叶片缺陷分割

## 5.1 当前状态

目标仓库 EGCIENet 适配器没有真正加载模型；上游代码使用 SegFormer MiT-B3 骨干和边缘引导，但数据加载代码存在目录和返回值不一致，不能直接假设可训练。

## 5.2 主数据：AEBIS

第一版将任务定义为**二值叶片缺陷分割**：缺陷像素=1，背景=0。这样能最大程度复用 AEBIS 标注并匹配上游模型。多类别分割作为后续扩展。

规范目录：

```text
artifacts/canonical/aebis_binary/
├── images/train/*.png
├── images/val/*.png
├── images/test/*.png
├── masks/train/*.png
├── masks/val/*.png
├── masks/test/*.png
├── edges/train/*.png
├── edges/val/*.png
├── edges/test/*.png
└── manifest.parquet
```

Mask 规范：单通道 `uint8`，仅允许值 0 和 255。Edge 必须由 mask 自动生成，不能成为独立人工真值：

```python
edge = morphological_gradient(mask, kernel=3)
```

## 5.3 辅助数据：BladeSynth

把 Dent/Nick/Scratch/Corrosion 所有缺陷掩膜合为二值，用于预训练。训练采样建议真实 AEBIS : 合成 BladeSynth = 1:1 或 2:1，不按原始数量自然采样。

不得把 BladeSynth 测试作为最终测试。可设置：

- `synthetic_train`；
- `aebis_train/val/test`；
- `external_real_test`。

## 5.4 重写上游数据加载

不要修补上游硬编码 `./data_587/Train/`。新建统一 Dataset：

```python
class BinaryDefectSegDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return {
            "image": float_tensor_C_H_W,
            "mask": float_tensor_1_H_W,
            "edge": float_tensor_1_H_W,
            "sample_id": str,
            "source": str,
        }
```

训练脚本：

```bash
python -m aero_training.train.egcienet \
  --config training/configs/experiments/egcienet_aebis_binary.yaml
```

必须支持：

- MiT-B3 预训练权重；
- 352×352 复现配置；
- 512 或多尺度工程配置；
- BCE + Dice/IoU + edge loss；
- AMP；
- 断点恢复；
- 早停；
- 保存 best Dice 和 best IoU 两个 checkpoint；
- 完整随机种子。

## 5.5 验收

- Dice、IoU、precision、recall、boundary F1；
- 每种原始缺陷类型的二值分割指标；
- 小缺陷召回；
- 空 mask 图像误报率；
- 对 100 张叠加结果做可视化审查；
- 上游论文指标只能作为参考，不得在没有完全复现时宣称达到；
- 导出 `final.pth` 或模型注册表指定名称，并修改目标适配器真实加载。

---

# 6. 视觉模型四：SAM-Adapter 裂纹分割

## 6.1 正确任务定义

该资产应专门处理**裂纹**，而不是所有叶片缺陷。当前基础 SAM + 中心点提示不构成 SAM-Adapter。代码 Agent 必须实现 Adapter/LoRA 或采用经过审计的上游实现，并定义提示策略。

## 6.2 数据层次

### 一级：真实叶片裂纹

- AEBIS 中明确标为 crack 的多边形/掩膜；
- 自有孔探裂纹图，由专家逐像素标注；
- 这是最终微调和验证的核心。

### 二级：通用裂纹预训练

#### CrackSeg9k

- 官方代码：`https://github.com/Dhananjay42/crackseg9k`；
- 数据托管于 Harvard Dataverse，V4 为最新整理版本；
- 约 9,160-9,255 张，具体以下载 manifest 为准；
- 已统一二值掩膜；
- 来源表面多样但并非叶片。

#### DeepCrack

- 官方数据/代码：`https://github.com/yhlleo/DeepCrack`；
- 537 张 RGB；300 train、237 test；
- 手工二值掩膜；
- 非叶片域。

#### CFD

- 官方仓库：`https://github.com/cuilimeng/CrackForest-dataset`；
- 118 张道路图像，手工轮廓；
- 仅用于小型外部泛化验证。

## 6.3 禁止的错误映射

- BladeSynth `Scratch` 不自动等于 `Crack`；
- GC10/NEU 的 crease/crazing 等不能未经专家审查映射到叶片裂纹；
- 路面裂缝测试成绩不能宣称为叶片裂纹性能。

## 6.4 训练流程

1. 使用 CrackSeg9k + DeepCrack 训练 Adapter，学习细长裂纹形态；
2. 用 AEBIS crack 子集微调；
3. 用自有叶片裂纹数据最终微调；
4. 冻结大部分 SAM encoder，优先 ViT-B 降低计算量；
5. 提示策略至少支持：
   - 来自 SLF-YOLO crack bbox 的 box prompt；
   - 自动候选点；
   - 人工点/框；
6. 对无 prompt 自动模式必须明确其候选生成方法，不能默认中心点；
7. 输出 mask、置信度、提示来源。

## 6.5 数据格式

```text
images/<split>/<id>.png
masks/<split>/<id>.png
prompts/<split>/<id>.json
```

Prompt JSON：

```json
{
  "boxes_xyxy": [[x1, y1, x2, y2]],
  "points_xy": [[x, y]],
  "point_labels": [1],
  "prompt_source": "ground_truth_jitter|detector|manual"
}
```

训练期间对真值框做随机扰动，模拟检测器误差；测试必须分别报告 oracle prompt 和 detector prompt，不能只报告 oracle。

## 6.6 验收

- AEBIS/自有叶片裂纹 Dice、IoU、boundary F1；
- 细裂纹连通性和断裂率；
- detector prompt 下的端到端性能；
- 空图误报；
- 输出必须真实使用 Adapter 参数；
- checkpoint 应包含 adapter 权重与基础 SAM 版本/哈希。

---

# 7. 视觉模型五：TS-SAM

## 7.1 数据适配原则

TS-SAM 官方代码面向伪装目标、阴影、显著目标、息肉等分割任务，公开任务权重不是航空叶片权重。其作用应定义为“第二种叶片二值缺陷分割架构”，而不是直接下载通用权重后宣布可用。

**官方代码：** `https://github.com/maoyangou147/TS-SAM`。  
官方环境示例包含 Python 3.8、PyTorch 1.13、MMCV 1.7，建议单独容器化，不与现代 Ultralytics 环境混装。

## 7.2 推荐数据

- 主：AEBIS 二值缺陷；
- 辅：BladeSynth 二值缺陷；
- 外部：AeBAD 异常掩膜可在许可和格式允许时作外部测试，不用于盲目训练；
- 官方 COD/SD/SOD 等数据仅用于复现原代码，不作为叶片模型最终数据。

## 7.3 执行策略

1. 先用官方 task 配置跑通单批次，确认环境；
2. 新建 `BladeBinaryDataset`，不修改官方数据集类的语义；
3. 创建叶片配置，明确 backbone、输入尺寸和双流模块；
4. 使用官方 SAM checkpoint 初始化；
5. 训练 AEBIS+BladeSynth；
6. 与 EGCIENet 在同一 split、同一指标下比较；
7. 如果无法显著优于 EGCIENet 或推理成本过高，不晋升为默认模型。

## 7.4 验收

同 EGCIENet，并额外记录：显存、参数量、FPS、端到端延迟。仅在真实 AEBIS 测试上优于或补充 EGCIENet 时保留。

---

# 8. 信号模型一：WCamba 轴承故障分类

## 8.1 必须修复的代码错配

上游 WCamba：

- 模型类实际为 `mambaModel`；
- 输入窗口为 1024 点；
- 针对不同数据集动态输出 10/14/5/7/3 类；
- 上游脚本使用 50% 重叠窗口和随机切分，存在泄漏风险。

目标适配器：

- 尝试导入不存在的 `models.wcamba.WCambaModel`；
- 截取 2048 点；
- 固定四类 `inner_race/outer_race/ball/cage`；
- 缺少 `normal`。

训练前必须重写适配器和模型配置。

## 8.2 第一主数据：CWRU

**官方位置：** Case Western Reserve University Bearing Data Center。  
**官方页面：** `https://engineering.case.edu/bearingdatacenter`  
**格式：** MATLAB `.mat`。  
**通道：** drive-end、fan-end、base、RPM 等，具体取决于文件。  
**类别：** normal、inner race、outer race、ball；没有 cage 类。  
**采样率：** 常用 12 kHz 和 48 kHz 数据；不同位置/实验文件不同。  
**负载：** 多种马力/转速工况。

### 两套训练任务

#### A. 上游复现 10 类

- normal 1 类；
- ball 3 种故障尺寸；
- inner 3 种；
- outer 3 种；
- 总计 10 类。

用于验证 WCamba 实现与上游一致，不直接接入生产类别。

#### B. 系统接入 4 类

```json
{
  "0": "normal",
  "1": "inner_race",
  "2": "outer_race",
  "3": "ball"
}
```

把不同故障尺寸合并为部位类。不要创建 cage。

### 读取规范

- 使用 `scipy.io.loadmat`；
- 不按变量顺序猜通道；通过键名识别 `DE_time`、`FE_time`、RPM；
- 第一版只用 12 kHz drive-end，减少域混合；
- 每个原始 MAT 文件生成一个 `record_id`；
- 窗口长度 1024；
- 训练可 50% 重叠，验证/测试不重叠；
- 标准化参数只从训练记录计算；
- 保存 sample_rate、load、fault_size、fault_location。

### 防泄漏划分

禁止把同一 MAT 文件窗口随机切到不同集合。推荐：

- 按原始 fault instance + load 分组；
- 至少保留一个负载工况作为外部工况测试；
- 或按故障尺寸分组，测试未见尺寸；
- 报告同工况和跨工况两套结果。

## 8.3 第二数据：Paderborn

**官方位置：** `https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter`  
**特点：** 同步振动和电机电流；包含健康、人工损伤和加速寿命产生的真实损伤；多运行工况。  
**标签：** 必须读取官方元数据，不能仅凭文件名前缀猜全部故障细节。

建议单独训练模型：

```json
{"0":"healthy", "1":"inner_race", "2":"outer_race"}
```

不建议把 Paderborn 与 CWRU 直接拼接成一个四类训练集，因为传感器、采样率、试验台、负载和标签体系差异大。更稳妥的是：

- 两个独立模型；
- 模型注册表中记录 `supported_dataset_domain`；
- 或在有足够数据时做领域自适应。

## 8.4 XJTU-SY 的正确用途

**官方位置：** `https://biaowang.tech/xjtu-sy-bearing-datasets/` 或作者 GitHub。  
**规模：** 15 个轴承、3 个工况、完整运行到失效；每分钟采样一次，每个文件 32,768 点，两通道，25.6 kHz。  
**用途：** 退化检测、异常评分、轴承 RUL。  
**不建议：** 把整条寿命序列的所有早期窗口都标为最终故障类型来训练平衡四类分类器，因为早期健康阶段会产生标签噪声和身份泄漏。

## 8.5 WCamba 训练与产物

模型配置：

```yaml
model_id: wcamba_cwru_4class
input_length: 1024
channels: 1
sample_rate_hz: 12000
classes: [normal, inner_race, outer_race, ball]
normalization: train_global_zscore
```

训练：

- 先严格复现上游 10 类；
- 再训练 4 类接入版；
- 使用按 record 分组的数据加载器；
- 记录 macro-F1、balanced accuracy、每类 recall、跨负载性能；
- 混淆矩阵必须包含 normal；
- 保存 `state_dict` 而不是整个 pickle 模型对象；
- 额外导出 TorchScript 或 ONNX 进行加载测试。

验收：

- 4 类内部测试 macro-F1 建议至少 0.90，但跨工况指标必须单独报告；
- 所有测试样本 group 与训练为零交集；
- 适配器输入长度、归一化和类序一致；
- 缺少 RPM/采样率时不进行故障特征频率解释；
- 输出不得含不存在的数据类 `cage`。

---

# 9. 信号模型二：FaultSense LSTM Autoencoder + RUL

## 9.1 上游真实情况

FaultSense 上游是 PyTorch，不是 TensorFlow。其仓库已经包含或说明：

- LSTM autoencoder；
- RUL head；
- FD001/FD003 `.pt` 权重；
- Ridge `.pkl` 基线；
- C-MAPSS 数据处理；
- 约 30-cycle 输入窗口；
- 多传感器输入；
- 验证误差阈值。

目标适配器期待 `.h5`，必须完全重写。

**上游位置：** `https://github.com/momo-2609/FaultSense-LSTM-Anomaly-Detection-on-NASA-CMAPSS`

## 9.2 主数据：NASA C-MAPSS

**官方入口：** `https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data`  
**官方 ZIP：** NASA Open Data 页面列出的 `CMAPSSData.zip`。  
**格式：** 空格分隔文本，无表头，26 列：

```text
unit_id, cycle,
op_setting_1, op_setting_2, op_setting_3,
sensor_1 ... sensor_21
```

测试集的 `RUL_FD00x.txt` 每行对应一个测试发动机最后观测点到失效的剩余周期。

子集：

| 子集 | 训练轨迹 | 测试轨迹 | 工况 | 故障模式 |
|---|---:|---:|---:|---:|
| FD001 | 100 | 100 | 1 | 1，HPC 退化 |
| FD002 | 260 | 259 | 6 | 1，HPC 退化 |
| FD003 | 100 | 100 | 1 | 2，HPC + Fan |
| FD004 | 248 | 249 | 6 | 2，HPC + Fan |

下载后必须验证列数=26、unit/cycle 单调性、每个 unit 最终周期、测试 RUL 行数与 unit 数一致。

## 9.3 FaultSense 复现方案

### 第一步：加载上游权重进行 smoke test

- 固定上游 commit；
- 读取其 checkpoint metadata；
- 精确复现其传感器选择、归一化、窗口长度和 latent size；
- 在上游示例输入上验证输出；
- 不能把上游权重直接用于目标适配器旧的 7 传感器接口。

### 第二步：重训

优先 FD001 与 FD003，保持上游 30-cycle 窗口。数据 split：

- 从训练 unit 中按 unit_id 划 70/15/15；
- 官方 test 仅用于最终测试；
- autoencoder 的正常训练区间必须定义，例如每个训练 unit 的早期 30%-40% 周期；
- 验证集正常重构误差确定阈值；
- 异常标签不能仅通过测试 RUL 反推后再混入训练阈值。

### 传感器

代码 Agent 必须从上游代码提取精确 feature list，并写入 `preprocessing.json`。常见 C-MAPSS 处理会剔除近常数通道，但不得凭记忆硬编码后不验证。输出必须包括原始传感器索引顺序。

### 归一化

- FD001/FD003：train-only MinMax 或 z-score；
- FD002/FD004：必须按运行工况分组或聚类后归一化；
- scaler 只拟合训练 unit；
- 保存 scaler 参数；
- 测试超范围值不剪裁或剪裁策略要明确。

## 9.4 接入设计

新适配器输入建议：

```json
{
  "sequence": [[...14 sensors...], ... 30 rows],
  "sensor_names": ["s2", "s3", "..."],
  "dataset_profile": "cmapss_fd001",
  "operating_settings": [[...], ...]
}
```

输出：

```json
{
  "anomaly_score": 0.0,
  "threshold": 0.0,
  "is_anomaly": false,
  "rul_cycles": 0.0,
  "status": "success",
  "model_version": "..."
}
```

不要强行把 C-MAPSS 匿名传感器映射成目标系统中的 `exhaust_temp` 等真实物理量。

## 9.5 验收

- 复现上游 FD001/FD003 指标趋势；
- 重构误差阈值仅由验证正常数据得到；
- RUL 至少报告 RMSE、MAE、NASA asymmetric score；
- 对输入传感器顺序变化必须报错；
- 30×N 和 N×30 维度混淆必须单测；
- 上游 checkpoint 仅 smoke test，最终需有本仓库可复现训练日志和权重。

---

# 10. 信号模型三：Isolation Forest 多传感器异常检测

## 10.1 公开数据的限制

目标代码要求七个具名传感器：排气温度、三轴/双轴振动、轴承温度、入口压力、润滑油压力、燃油流量。公开 C-MAPSS 的传感器主要以匿名索引发布，不能诚实地一一改名。没有公开数据能够完全匹配这七个字段并代表目标发动机。

因此：

- **生产版 Isolation Forest 必须使用自有正常遥测；**
- C-MAPSS/N-CMAPSS 只能训练研究代理模型；
- UCI 地面燃气轮机 CO/NOx 数据可用于测试运行工况聚类和异常管线，但无故障标签，且不是航空发动机。

## 10.2 推荐真实数据格式

```text
telemetry.parquet
engine_id                string
timestamp                timestamp
operating_regime         string|int
exhaust_temp_c           float
vibration_x_g            float
vibration_y_g            float
bearing_temp_c           float
inlet_pressure_kpa       float
lube_oil_pressure_kpa    float
fuel_flow_kg_h           float
maintenance_event_id     nullable string
confirmed_fault          nullable string
quality_flags            int
```

要求：统一单位、校准信息、传感器缺失标记、启动/停车状态、工况和维修事件。

## 10.3 特征工程

不要直接把单时刻七维值交给 IF。以固定窗口生成：

- mean、median、std、MAD；
- min/max/range；
- RMS、kurtosis、skewness；
- slope、first difference；
- vibration crest factor；
- 缺失率；
- 工况 one-hot 或每工况独立模型。

训练只用专家确认的健康期。每个发动机型号/传感器版本/工况应建立独立 profile。

## 10.4 模型保存

使用 sklearn Pipeline：

```python
Pipeline([
  ("imputer", SimpleImputer(...)),
  ("scaler", RobustScaler()),
  ("iforest", IsolationForest(...))
])
```

保存：

- `model.joblib`；
- `feature_schema.json`；
- `sensor_schema.json`；
- `thresholds.json`；
- `training_periods.parquet`；
- 校准报告。

阈值不要直接依赖 `contamination`。用独立健康验证期的 score 分位数和少量已确认事件共同校准。

## 10.5 研究代理数据

### C-MAPSS/N-CMAPSS

选择若干传感器，但名称必须保持 `s1...s21` 或官方字段，不映射成物理名。以早期寿命作为近似健康期，晚期作为弱异常评估。

### UCI Gas Turbine CO/NOx

- 36,733 条；
- 11 个小时聚合传感量；
- 用途是排放回归，不是故障诊断；
- 可用于无监督管线 smoke test 和工况漂移演示；
- 不得作为航空发动机故障模型证据。

## 10.6 验收

- 未加载已拟合模型时必须 `unavailable`；
- 特征 schema 不匹配立即失败；
- 独立健康期假阳性率；
- 已知维修事件前的提前量；
- 按工况、季节、机号分层；
- 不得把研究代理模型标记为 production。

---

# 11. RUL 模型一：CNN-LSTM

## 11.1 主数据与目标接口

主数据为 C-MAPSS。目标仓库当前 `CNNLSTMRULPredictor` 是线性趋势占位，必须改成真正 Conv1d + LSTM 回归模型。

推荐输入：

```text
[batch, sequence_length=50, num_features]
```

推荐基础结构：

```text
Conv1d -> BatchNorm -> ReLU -> Conv1d -> ReLU -> LSTM -> MLP -> RUL
```

## 11.2 RUL 标签

训练集真实最终周期已知：

```python
raw_rul = max_cycle_of_unit - cycle
rul = min(raw_rul, 130)
```

RUL cap 必须配置化并与当前 `CMAPSSDatasetAdapter` 的 130 对齐。不得训练 125 cap、推理却按 130 解释而不记录。

## 11.3 数据准备

- 26 列解析为 `unit_id/cycle/op1-op3/s1-s21`；
- 删除或保留低方差传感器由训练数据统计决定；
- 保存 feature list；
- FD002/FD004 使用工况归一化；
- sequence length 50；
- 短序列左侧复制第一行或 mask padding，策略写入配置；
- train stride 可 1-5；
- validation/test 每 unit 使用末窗口评估官方 RUL，同时可报告全轨迹窗口性能。

## 11.4 划分

- 官方 test 不参与调参；
- 训练 unit 分组 70/15/15；
- 对每个随机种子保存 unit 列表；
- 不允许行级随机 split。

## 11.5 不确定度

当前代码输出固定 ±15 cycles 不可接受。至少采用一种：

- 5 个独立种子 ensemble；
- MC dropout；
- quantile regression 输出 P10/P50/P90；
- heteroscedastic Gaussian head。

推荐 quantile loss + ensemble，输出校准后的区间覆盖率。

## 11.6 评估与门槛

每个 FD 子集报告：

- RMSE；
- MAE；
- NASA asymmetric score；
- 每 unit 误差分布；
- 早期/中期/晚期误差；
- 预测区间覆盖率和宽度；
- 与 last-value、线性趋势、Ridge、普通 LSTM 比较。

晋升门槛：

1. 比当前线性趋势基线 RMSE 和 NASA score 至少改善 10%；
2. FD001 与 FD003 的最终 RMSE 建议不高于 20 cycles，否则保留 experimental；
3. 多种子均值和标准差完整；
4. 无 unit 泄漏；
5. 模型在 CPU 上可加载推理；
6. 所有预处理参数随权重发布。

这些门槛是工程门禁，不等于真实机队有效。

---

# 12. RUL 模型二：ChangePoint-LSTM

## 12.1 官方资源

**官方代码：** `https://github.com/en-research/ChangePoint-LSTM`  
**论文任务：** 在可变工况下先检测每台设备的退化变点，再利用变点改善 RUL 标注和 LSTM 估计。  
**最匹配数据：** FD002、FD004，因为包含六种运行工况；FD001/FD003 可用于简化验证。

## 12.2 执行步骤

1. 固定官方代码 commit；
2. 复现官方数据处理和结果；
3. 审计变点检测是否使用了测试未来信息；
4. 把变点检测和 RUL 模型拆成两个可测试组件；
5. 输出每个 unit 的 change point、控制限、监测统计；
6. 仅在检测到退化后触发 RUL，未检测到时返回健康状态而非虚假剩余寿命；
7. 接入目标 `ChangePointLSTRUL`，替换分段线性占位。

## 12.3 标签与评估

- 变点没有公开绝对真值时，必须说明它是算法估计；
- 可利用 N-CMAPSS 更丰富的退化变量作辅助验证；
- 报告 change point 稳定性、不同种子的偏移、RUL 改善；
- 与不使用变点的同结构 LSTM 做严格消融。

## 12.4 晋升条件

只有当变点版在 FD002/FD004 的同一 split 上稳定优于无变点版，且不存在未来信息泄漏时，才接入默认计划。

---

# 13. PINN Fleet Prognosis

## 13.1 数据现实

目标中的 PINN 舰队寿命模型当前未实现。公开的 PML-UCF PINN 资源主要面向风电轴承或腐蚀疲劳，不能直接成为航空发动机舰队模型。公开 C-MAPSS/N-CMAPSS 可以支持研究，但缺少目标发动机型号的真实材料、载荷、维护和机队差异参数。

## 13.2 推荐研究数据

- N-CMAPSS：复杂飞行工况和退化轨迹；
- C-MAPSS：基线；
- 自有机队数据：发动机序号、飞行循环、环境、维护、部件更换、传感器、失效/拆检结果；
- 物理参数：热力学约束、损伤演化方程、边界条件。

## 13.3 不能自动完成的部分

代码 Agent 可以建立训练框架和研究模型，但以下内容必须由领域工程师提供：

- 物理方程是否适用于特定发动机；
- 状态变量及单位；
- 边界/初始条件；
- 机队层随机效应；
- 维修导致的状态重置；
- 适航和维修阈值。

没有这些信息时，产物只能标记 `research_only`。

---

# 14. 物理/工程模型：FDPP、Paris、S-N、pyLife、py-fatigue

这些不是普通监督学习，不应寻找“一个公开图像数据集”来训练。

## 14.1 必要输入

### 裂纹扩展

- 初始裂纹尺寸与测量不确定度；
- Paris 常数 C、m；
- ΔK threshold；
- 断裂韧度 K_IC；
- 应力比 R；
- 温度、环境、频率；
- 几何修正因子 Y(a)；
- 实际载荷谱和雨流计数；
- 检查间隔与检出概率。

### S-N/疲劳寿命

- 对应材料、表面状态、温度、应力比的 S-N 曲线；
- 均值应力修正；
- 尺寸、缺口、表面和可靠度系数；
- 部件应力历史。

## 14.2 推荐软件

- pyLife：`https://github.com/boschresearch/pylife`；
- py-fatigue：`https://owi-lab.github.io/py_fatigue/`。

## 14.3 数据文件规范

```yaml
material_id: ...
material_batch: ...
temperature_c: ...
stress_ratio_R: ...
paris:
  C: ...
  m: ...
  deltaK_threshold: ...
  K_IC: ...
geometry:
  type: ...
  parameters: ...
  Y_model: ...
provenance:
  standard: ...
  report_id: ...
  approved_by: ...
```

缺任何关键参数时返回 `unavailable`，不得用网上通用常数填充后输出部件寿命。

---

# 15. 数据下载与转换脚本清单

代码 Agent 应至少实现：

```text
training/scripts/download/
├── download_aebad.py
├── download_ca2.py
├── download_bladesynth.py
├── download_aebis.py
├── download_neu_det.py
├── download_gc10_det.py
├── download_crackseg9k.py
├── download_deepcrack.py
├── download_cwru.py
├── download_paderborn.py
├── download_xjtu_sy.py
├── download_cmapss.py
└── download_n_cmapss.py

training/scripts/convert/
├── prepare_aebad.py
├── labelme_to_masks_boxes.py
├── bladesynth_masks_to_yolo.py
├── crack_datasets_to_binary.py
├── cwru_mat_to_parquet.py
├── paderborn_to_parquet.py
├── xjtu_sy_to_parquet.py
├── cmapss_to_parquet.py
└── telemetry_to_if_features.py
```

每个脚本统一参数：

```bash
--output-root
--cache-root
--force
--verify-only
--offline
--workers
--seed
```

统一退出码：

- 0 成功；
- 2 需要人工下载/授权；
- 3 校验失败；
- 4 许可证未确认；
- 5 数据格式与预期不符。

---

# 16. 训练脚本与配置

建议命令：

```bash
python -m aero_training.train.anomaly_mmr --config ...
python -m aero_training.train.slf_yolo --config ...
python -m aero_training.train.egcienet --config ...
python -m aero_training.train.sam_adapter --config ...
python -m aero_training.train.ts_sam --config ...
python -m aero_training.train.wcamba --config ...
python -m aero_training.train.faultsense --config ...
python -m aero_training.train.isolation_forest --config ...
python -m aero_training.train.cnn_lstm_rul --config ...
python -m aero_training.train.changepoint_lstm --config ...
```

每个训练脚本必须支持：

- `--dry-run`：只加载 2 批数据；
- `--smoke-test`：1 epoch，小样本；
- `--resume`；
- `--device`；
- `--seed`；
- `--deterministic`；
- `--output-dir`；
- 配置和环境快照；
- W&B/MLflow 可选但不得强依赖外部服务；
- stdout + JSONL 日志；
- 中断后可恢复；
- 最佳权重与最后权重分开。

---

# 17. 自动化验收测试

## 17.1 数据测试

- 图像全部可解码；
- mask 与 image 同尺寸；
- mask 值合法；
- bbox 在图内；
- 类别 ID 连续且与 class_map 相同；
- 每类样本数和实例数报告；
- group split 无交集；
- 近重复跨 split 扫描；
- 时间序列 unit 不跨 split；
- 标准化只拟合 train；
- 下载 manifest 与 SHA256 完整。

## 17.2 模型测试

- 权重可在无训练代码环境加载；
- CPU 推理；
- GPU 推理；
- batch=1 和 batch>1；
- 异常维度、NaN、空图、全零信号；
- 类别序一致；
- golden input 输出在允许容差内；
- 无权重时返回 `unavailable`；
- baseline 明确标记 `degraded`。

## 17.3 端到端测试

对每个模型：

```text
原始输入 -> 目标系统 API -> 模型适配器 -> 真实权重 -> 标准化输出
```

检查：

- model provenance 出现在 API 响应；
- 结果中无本地绝对路径和秘密；
- 延迟、峰值内存；
- 同一输入重复运行稳定；
- 版本切换不会错用旧 scaler/class_map。

---

# 18. 模型晋升状态

模型注册表只允许：

```text
stub
baseline
experimental
validated_public_domain
validated_target_domain
production_candidate
retired
```

晋升规则：

- 只有下载/训练/测试/接入完整，才能从 `stub` 到 `experimental`；
- 公开数据独立测试通过，才能 `validated_public_domain`；
- 自有目标域盲测通过，才能 `validated_target_domain`；
- 工程专家、质量和安全流程完成后，才可 `production_candidate`；
- 任何公开数据模型都不得直接标记 production。

---

# 19. 推荐资源与算力规划

以下是工程估算，不是保证：

| 模型 | 建议 GPU | 数据空间 | 训练难度 |
|---|---|---:|---|
| PatchCore/轻量异常检测 | 8-16 GB | 20-50 GB | 低-中 |
| MMR AeBAD | 16-24 GB | 30-80 GB | 中-高 |
| SLF-YOLO 640 | 12-24 GB | 80-150 GB | 中 |
| EGCIENet 352 | 12-24 GB | 30-80 GB | 中 |
| SAM-Adapter ViT-B | 16-24 GB | 80-150 GB | 高 |
| TS-SAM | 24 GB 以上或多卡 | 80-150 GB | 高 |
| WCamba | 8-16 GB | 20-50 GB | 中 |
| FaultSense | 8-16 GB | 10-30 GB | 中 |
| CNN-LSTM RUL | 8-16 GB | 10-30 GB | 中 |
| N-CMAPSS/PINN | 24 GB 以上、较大内存 | 100 GB 以上 | 高 |

所有环境应容器化。TS-SAM 等旧依赖单独镜像，避免 MMCV、CUDA 和 PyTorch 冲突。

---

# 20. 代码 Agent 的最终执行清单

代码 Agent 必须逐项产生证据文件，而不是只输出“完成”：

## Phase 0

- [ ] 记录目标仓库当前 commit；
- [ ] 建立 model registry；
- [ ] 修复 WCamba 输入/类别/导入；
- [ ] 修复 FaultSense PyTorch 加载；
- [ ] 修复 IF 已拟合检查；
- [ ] 为所有模型增加 provenance；
- [ ] 所有单测和 CI 通过。

## Phase 1A：数据

- [ ] 下载 AeBAD、BladeSynth、AEBIS、CWRU、C-MAPSS；
- [ ] 生成 SHA256、许可证快照、样本统计；
- [ ] 转换统一格式；
- [ ] 人工审计包：每数据集至少 100 张/窗口可视化；
- [ ] split 泄漏审计通过。

## Phase 1B：训练

- [ ] MMR/PatchCore；
- [ ] SLF-YOLO；
- [ ] EGCIENet；
- [ ] WCamba 10 类复现；
- [ ] WCamba 4 类接入；
- [ ] FaultSense smoke test 与重训；
- [ ] CNN-LSTM。

## Phase 1C：接入

- [ ] 权重目录符合统一格式；
- [ ] 目标适配器真实加载；
- [ ] golden samples；
- [ ] API 端到端；
- [ ] 不再静默使用 Sobel/Canny/线性趋势；
- [ ] 生成总体验收报告。

## Phase 2

- [ ] CrackSeg9k/DeepCrack；
- [ ] SAM-Adapter；
- [ ] TS-SAM；
- [ ] Paderborn；
- [ ] ChangePoint-LSTM；
- [ ] 自有正常遥测 IF。

---

# 21. 必须交付的最终报告

`reports/final_training_report.md` 至少包含：

1. 仓库 commit；
2. 数据下载状态和许可证；
3. 每个数据集实际样本数，而不是论文宣称数；
4. 数据转换错误和丢弃样本；
5. split group 列表；
6. 每个模型配置、参数量、训练时间和环境；
7. 完整指标和置信区间；
8. 跨域指标；
9. 失败模型与原因；
10. 权重 SHA256；
11. 接入测试；
12. 当前最高成熟度；
13. 下一步需要的自有数据。

不得使用“模型训练成功”这种无证据表述。应写成：

```text
模型 X 在数据集 Y 的固定 split Z 上完成训练；
权重 SHA256 为 ...；
测试指标为 ...；
适配器端到端测试通过 ... 个样本；
尚未在真实目标机型上验证，因此状态为 validated_public_domain。
```

---

# 22. 主要来源清单

以下来源应由代码 Agent 在执行时再次保存页面快照和访问日期。

| 编号 | 来源 | 地址 | 用途 |
|---|---|---|---|
| S01 | 目标仓库 | https://github.com/WYZAAACCC/Aero-engine-Damage-Detection-System | 代码接口与集成目标 |
| S02 | MMR/AeBAD 官方 | https://github.com/zhangzilongc/MMR | AeBAD 数据与 MMR |
| S03 | CA² 官方 | https://github.com/changniu54/CA2 | 多发动机叶片图像 |
| S04 | BladeSynth 数据 DOI | https://doi.org/10.6084/m9.figshare.28658603 | 合成叶片图像与掩膜 |
| S05 | BladeSynth 论文 | https://www.nature.com/articles/s41597-025-05563-y | 数据规模、类别、验证 |
| S06 | EGCIENet/AEBIS | https://github.com/Newbiejy/EGCIENet_In-service-blade-defect-detection | AEBIS 与网络代码 |
| S07 | SLF-YOLO | https://github.com/zacianfans/SLF-YOLO | 检测网络上游 |
| S08 | NEU-DET 官方页 | http://faculty.neu.edu.cn/songkechen/zh/CN/zdylm/263270/list/index.htm | 通用钢表面检测 |
| S09 | GC10-DET 原作者仓库 | https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets | 通用金属检测 |
| S10 | SAM-Adapter crack 上游 | https://github.com/sky-visionX/CrackSegmentation | 通用裂纹 Adapter 参考 |
| S11 | CrackSeg9k | https://github.com/Dhananjay42/crackseg9k | 通用裂纹分割 |
| S12 | DeepCrack | https://github.com/yhlleo/DeepCrack | 通用裂纹分割 |
| S13 | CFD | https://github.com/cuilimeng/CrackForest-dataset | 裂纹外部测试 |
| S14 | TS-SAM | https://github.com/maoyangou147/TS-SAM | TS-SAM 上游 |
| S15 | CWRU Bearing Data Center | https://engineering.case.edu/bearingdatacenter | 轴承分类主数据 |
| S16 | Paderborn Bearing Data Center | https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter | 轴承跨工况数据 |
| S17 | XJTU-SY | https://biaowang.tech/xjtu-sy-bearing-datasets/ | 轴承全寿命数据 |
| S18 | FaultSense | https://github.com/momo-2609/FaultSense-LSTM-Anomaly-Detection-on-NASA-CMAPSS | PyTorch 权重与处理 |
| S19 | NASA C-MAPSS | https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data | 发动机 RUL 主数据 |
| S20 | N-CMAPSS 论文/数据说明 | https://phm-datasets.s3.amazonaws.com/NASA/17.+N-CMAPSS.zip | 复杂工况研究；执行时核验官方落地页 |
| S21 | ChangePoint-LSTM | https://github.com/en-research/ChangePoint-LSTM | 变点 RUL 官方代码 |
| S22 | UCI Gas Turbine | https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set | 工况/异常辅助 |
| S23 | pyLife | https://github.com/boschresearch/pylife | 疲劳寿命工程算法 |
| S24 | py-fatigue | https://owi-lab.github.io/py_fatigue/ | Paris/循环计数/裂纹增长 |

---

# 23. 最终推荐

要最快得到真正能被当前系统正常加载和使用的权重，不要同时追求全部论文模型。优先顺序应为：

1. **AeBAD + MMR/PatchCore**：完成无监督叶片异常检测；
2. **AEBIS + EGCIENet**：完成真实叶片二值缺陷分割；
3. **BladeSynth + AEBIS + SLF-YOLO**：完成多类缺陷定位；
4. **CWRU + 修正后的 WCamba**：完成正常/内圈/外圈/滚动体四类诊断；
5. **C-MAPSS + FaultSense**：完成异常与 RUL 上游复现；
6. **C-MAPSS + CNN-LSTM**：替换线性占位；
7. 再实施 SAM-Adapter、TS-SAM、Paderborn、ChangePoint；
8. Isolation Forest、PINN 和物理寿命模型必须等待自有遥测、材料和载荷数据。

公开数据训练完成后，系统可以达到“软件可用、公开基准有效”的水平，但要达到真实航空发动机可用，仍必须建立自有孔探和遥测盲测集，并由发动机工程专家参与标签、阈值和失效模式审核。
