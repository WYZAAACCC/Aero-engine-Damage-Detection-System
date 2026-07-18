# Aero-engine-Damage-Detection-System 深度实现审计与修复指导

> 审计对象：`WYZAAACCC/Aero-engine-Damage-Detection-System`，`main` 分支公开源码  
> 审计日期：2026-07-18  
> 审计性质：源码级架构审计、关键路径静态检查、局部可执行对抗测试  
> 结论适用范围：当前公开仓库版本；不代表对未公开模型权重、私有数据、OEM 手册授权内容或外部服务的认证

---

## 1. 执行摘要

### 1.1 总体结论

该项目**不是完全空壳**。它包含两个质量差异很大的部分：

1. `seekflow` 通用智能体运行时具备真实的 LLM 对话、工具注册、工具调用循环、策略控制、沙箱和审计等基础能力；
2. `aero_diag` 航空发动机领域层拥有较完整的数据模型、资产清单、状态机、计划模型、证据模型和 API 外观，但**真正把 Agent、计划编译、资产执行、知识检索、证据验证和专家审批串成闭环的实现尚未完成**。

因此，当前系统更准确的定位是：

> **可信 Agent 运行时之上的航空发动机诊断架构原型和算法适配演示，而不是已经实现的安全关键诊断系统。**

当前公开代码不应被用于真实发动机放行、继续服役、停飞、维修、复检周期或剩余寿命决策。其主要风险并非单纯“精度不足”，而是系统中存在多条路径会将启发式方法、随机初始化模型、通用模型或缺乏适用域验证的结果标记为 `success`，随后被规则层当成有效工程证据。

### 1.2 能力真实性分级

| 能力 | 当前判断 | 说明 |
|---|---|---|
| SeekFlow 通用 LLM 工具调用 | 真实但通用 | `DeepSeekAgent` 可注册工具并进入 ToolRuntime 循环 |
| 航空领域 Agent | 基本未实现 | `aero_diag/agents` 只有角色元数据，没有领域 Agent 控制器、工具绑定和闭环执行 |
| Agent 自动规划 | 演示级 | 有 `PlanProposal/PlanCompiler`，但测试 Agent 不调用它，编译器校验也不完整 |
| 计划执行引擎 | 未实现 | `/runs` 明确返回 `not_implemented` |
| 状态质量门 | 伪约束 | 门条件是文字描述；未注册检查器时仍允许迁移并记录为已满足 |
| 工程资产调用框架 | 部分真实 | Runner 会调用 Python 实现，但没有强制 Schema、适用域、状态、输出语义和模型包验证 |
| 视觉检测资产 | 多数为替代/伪实现 | 通用 ResNet、通用 COCO YOLO、中心点 SAM 和边缘启发式被包装成领域模型 |
| 信号检测资产 | 高风险替代实现 | 无权重时可使用随机初始化 CNN/LSTM，或退化为任意阈值趋势算法 |
| 损伤几何表征 | 不可靠 | 框格式歧义可产生负面积；未真正读取 mask；没有骨架长度实现 |
| RUL/裂纹扩展 | 演示级 | CNN-LSTM 实际为线性趋势；Paris 模型使用危险默认值并忽略部分参数 |
| 数据质量校验 | 表面化 | 非法数值、非法单位、空数据可以通过；不做真实量纲和适用性校验 |
| 专家知识检索 | 静态小型知识表 | 29 条手写字典；不是文档 RAG；多数没有页码、版本、修订号和可验证引用 |
| 专家审批 | 伪实现 | 任意字符串可充当审核人；无认证、签名验证、资质校验和证据包完整性校验 |
| 航空领域自动化测试 | 基本缺失 | 主测试目录主要覆盖 SeekFlow；领域脚本位于根目录且没有断言 |

### 1.3 最重要的审计结论

系统当前的最大问题不是“Agent 不够聪明”，而是以下四个控制面没有形成可信闭环：

- **Agent 生成的内容没有被强制转换并验证为结构化计划；**
- **计划没有由真实执行引擎执行和追踪；**
- **工具返回 `success` 不代表结果在工程语义上有效；**
- **知识、规则与专家签署没有形成可验证的安全边界。**

---

## 2. 审计方法和边界

本次审计采用以下方法：

1. 检查仓库目录、包配置、Agent、编排、服务、API、资产清单、资产实现、知识库和测试；
2. 沿任务创建、计划提出、计划编译、资产运行、知识检索、风险决策、专家审批和归档路径追踪调用关系；
3. 将代码注释和设计文档中的承诺与实际函数行为逐项对照；
4. 对计划编译、状态机、视觉基线、几何测量、数据质量、知识检索和规则决策构造对抗输入并本地执行；
5. 区分“运行框架真实”“算法名义存在”“算法工程有效”和“安全关键可使用”四种完全不同的状态。

本次没有调用真实 DeepSeek API，也没有逐页核验知识目录中的全部 PDF/DOCX，因此不会评价大模型在线回答质量或 OEM 手册条款本身是否真实。审计重点是：**公开源码是否能够保证 Agent 真正调用领域工具、工具是否正确处理数据、结果是否被有效校验，以及系统是否可能制造虚假可信度。**

---

## 3. Agent 实现审计

## 3.1 SeekFlow Agent 本身不是空壳

`src/seekflow/agent/agent.py` 中的 `DeepSeekAgent` 具备：

- 工具注册；
- ToolRuntime 创建；
- 多步工具调用；
- 工具调用结果记录；
- ReAct、Plan-Solve、Reflection 外观；
- 文件、网络、Python、SQLite 权限显式开启；
- 费用、上下文和工具调用诊断信息。

因此，不能简单说整个仓库的 Agent 都是假的。**通用 Agent 运行时有真实实现。**

但这个事实不能证明航空发动机诊断 Agent 已实现，因为领域系统没有把这些能力正确接上。

## 3.2 航空领域 Agent 只有角色声明，没有控制器

`src/aero_diag/agents/roles.py` 定义了多个 `AgentRole`：

- 任务协调；
- 数据质量；
- 检测；
- 损伤表征；
- 可靠性；
- 证据融合；
- 专家协作。

每个角色包含提示词、允许工具名称、禁止行为和确定性支撑服务名称，但当前目录没有看到：

- 将角色实例化成 `DeepSeekAgent` 的工厂；
- 将 `allowed_tools` 转成真实 Tool 注册的绑定器；
- 检查 Agent 是否只调用允许工具的运行时策略；
- 将 Agent 输出解析成 `PlanProposal` 的结构化解析器；
- 将 PlanProposal 交给 PlanCompiler 的应用服务；
- 计划执行、工具调用结果回传、重规划和终止控制器；
- 角色之间共享唯一事实源和证据图的编排器。

换言之，角色定义目前是**设计元数据**，不是已经运行的多 Agent 系统。

## 3.3 所谓 Agent 集成测试没有连接领域工具

根目录 `test_agent_integration.py`：

- 创建一个通用 `DeepSeekAgent`；
- 用“20 年 CFM56 维修经验”作为 `backstory`；
- 只调用 `agent.with_default_tools()`；
- 默认工具是计算、CSV 文本解析、实体抽取和文本分类等通用工具；
- 没有注册 `search_engineering_assets`、`run_engineering_asset`、`retrieve_domain_knowledge`、`compile_plan`、`validate_evidence` 或 `request_expert_review`；
- 没有把知识库文档通过 `add_documents` 或向量库连接给 Agent；
- 没有强制结构化输出或引用证据。

因此，大模型输出主要依赖 Prompt 中的人设和模型预训练知识，而不是系统中的工程资产和知识库。

更严重的是，该脚本的 fallback 部分虽然调用了知识、风险和复检资产，却直接打印硬编码结论：

- `Severity: SEVERE`；
- `Risk Level: HIGH`；
- `Decision Code: C`；
- `Inspection Interval: IMMEDIATE — 25 cycles`。

这些结论并不是完整地从工具返回结构化结果推导出来的，属于演示输出。

## 3.4 E2E 脚本也不是 Agent 闭环测试

根目录 `test_e2e_pipeline.py`：

- 手工创建任务；
- 手工调用 `runner.execute()`；
- 手工迁移状态；
- 使用合成振动和人工画出的暗色条纹图像；
- 手工调用知识和规则；
- 最后自动推进到 `EXPERT_REVIEW → APPROVED → ARCHIVED`；
- 没有 Agent；
- 没有 PlanProposal/PlanCompiler 的真实计划；
- 没有真实审核人；
- 没有测试断言。

它证明的是“若逐行手工调用这些对象，代码可以输出一串结果”，而不是 Agent 能自主完成规划、调用、验证和审批。

此外，`pyproject.toml` 配置 `testpaths = ["tests"]`，两个根目录脚本通常不会被默认 `pytest` 执行。

## 3.5 Agent 规划的真实缺口

当前 Plan-Solve 方法只是：

1. 第一次调用 LLM 生成 3–5 步自然语言计划；
2. 第二次把自然语言计划再次放进 Prompt 要求执行。

这不是工程意义上的计划编译。真正的领域计划应当包括：

- 资产 ID 与版本；
- 输入 Artifact；
- 输出 Artifact Schema；
- 适用域；
- 参数；
- 随机种子；
- 依赖 DAG；
- 失败策略；
- 质量门；
- 审批节点；
- 证据要求；
- 资源和超时。

仓库虽然定义了这些模型的一部分，但没有让 Agent 使用并执行它们。

---

## 4. 计划编译与状态机审计

## 4.1 PlanCompiler 声称的校验没有完整实现

`PlanCompiler` 注释声称会检查：

- DAG 无环；
- 无孤立节点；
- 资产适用域；
- 输入 Schema；
- 质量门和审批；
- 参数与种子冻结；
- 阶段覆盖；
- 证据覆盖。

实际实现仅完成了：

- 检查依赖节点 ID 是否存在；
- 检查是否至少有一个入度为 0 的节点；
- 解析资产；
- 只允许 `validated/qualified`；
- 冻结版本；
- 复制 `requires_approval`；
- 检查四种阶段是否出现；
- 检查 evidence_requirements 非空。

没有实现：

- 完整拓扑排序和全图环检测；
- 孤立节点与不可达节点检查；
- 阶段顺序检查；
- `input_refs` 与 `depends_on` 一致性；
- 上游输出和下游输入 Schema 匹配；
- 参数 JSON Schema 校验；
- 资产适用域与任务部件、材料、工况匹配；
- 输出类型检查；
- 资源和安全策略注入；
- 证据需求与具体节点输出的覆盖关系；
- 高风险计划审批状态验证。

### 对抗测试结果

构造四个节点：

- 一个合法入口节点；
- 一个正常后续节点；
- 另外两个节点形成独立循环。

编译器返回成功并生成冻结摘要：

```text
CYCLE_ACCEPTED 6fb6cfa4ce0f7796 ['a', 'b', 'c', 'd']
```

原因是它只检查“是否存在至少一个入度为零节点”，并未确认所有节点都能被拓扑排序。

## 4.2 计划摘要不能证明计划完整性

`ExecutionPlan.compute_digest()` 只哈希 `nodes`，不包含：

- `task_id`；
- `evidence_requirements`；
- `required_stages`；
- `approval_required`；
- `approval_reason`；
- 编译器版本；
- 资产 Manifest digest；
- 任务约束；
- 知识/规则版本；
- 数据分类和执行策略。

对抗测试中，仅修改证据要求和审批要求，摘要完全相同：

```text
DIGESTS 0190e02c5b87607e 0190e02c5b87607e equal=True
```

这意味着攻击者或程序错误可以改变决策相关属性，而“冻结摘要”仍保持不变。

## 4.3 状态机的门条件没有执行

`_TRANSITION_GATES` 保存的是中文描述字符串。`transition()` 只有在外部显式调用 `register_gate_checker()` 后才真正执行检查。

若没有检查器：

- 门描述直接被加入 `preconditions_met`；
- 状态迁移成功；
- 系统看起来像完成了门检查。

`TaskService.transition_state()` 每次都新建状态机，只设置私有 `_current_state`，没有注册检查器，也没有传入领域上下文。

### 对抗测试结果

任务没有任何输入 Artifact，仍然可以从 `RECEIVED` 进入 `DATA_VALIDATION`：

```text
GATE_BYPASS data_validation
preconditions_met=['任务必须包含至少一个 input_artifact 引用']
preconditions_failed=[]
```

这属于审计语义上的反向记录：**没有检查，却记录为满足。**

## 4.4 状态历史不持久

`TaskService` 每次迁移创建新的 `TaskStateMachine`，迁移历史只存在于临时对象中，没有保存回任务或数据库。因此无法可靠回答：

- 谁在何时推进了状态；
- 当时检查了哪些证据；
- 哪个门失败过；
- 哪个 Agent 或人工批准了迁移；
- 是否发生过回退或重新执行。

## 4.5 执行引擎尚未实现

`src/aero_diag/api/routes/runs.py` 明确返回：

```json
{
  "run_id": "placeholder",
  "status": "not_implemented",
  "message": "Execution engine — pending P1 implementation"
}
```

因此当前 API 控制面不能提交、追踪或恢复真正的执行计划。

---

## 5. 工程资产运行框架审计

## 5.1 Runner 能调用代码，但不保证工程正确

`AssetRunner` 会：

1. 解析资产；
2. 检查少量状态；
3. 查找实现；
4. 调用 `validate_inputs()`；
5. 调用 `run()`；
6. 返回结果。

但它没有强制：

- Manifest 输入 Schema；
- Manifest 参数 Schema；
- Manifest 适用域；
- 输入单位；
- 数据来源和哈希；
- OOD；
- 模型权重 digest；
- 输出 Schema；
- 输出值域和物理不变量；
- Artifact 持久化；
- 证据项生成；
- 运行溯源完整性；
- 安全策略和资源限制；
- 高风险资产审批。

`ImplementationBase.validate_inputs()` 默认总是通过，`health_check()` 默认总是健康。

## 5.2 资产状态可以被直接绕过

PlanCompiler 只允许 `validated/qualified`，但 AssetRunner 只拒绝 `deprecated/retired`。

因此 Runner 可直接执行：

- `draft`；
- `candidate`；
- `rejected`。

只要调用者绕过 PlanCompiler，资产治理即失效。

## 5.3 实现加载错误被静默吞掉

默认实现注册器：

```python
try:
    inst = cls()
    registry[cls.asset_id] = inst
except Exception:
    pass
```

后果包括：

- 依赖缺失不在启动时暴露；
- 实现构造错误不记录；
- 资产可能悄悄消失；
- 系统没有健康状态和告警；
- 测试环境与生产环境能力可能不同。

## 5.4 `success` 的语义被严重滥用

当前实现常见模式：

1. 尝试加载真实模型；
2. 失败；
3. 自动使用简单启发式；
4. 仍返回 `status="success"`。

这把以下不同状态混为一谈：

- 已验证模型成功；
- 未验证替代模型成功；
- 基线启发式执行完毕；
- 结果仅适合调试；
- 结果可以进入安全决策。

工程系统必须区分“程序运行成功”和“结果具有决策有效性”。

---

## 6. 视觉检测资产审计

## 6.1 CA² 异常检测器存在恒零异常分数

在“完整模型”路径中：

```python
score = float(np.linalg.norm(feat - feat))
```

任何向量减去自身都为零，因此：

- `score` 恒为 0；
- `anomaly_detected` 恒为 False；
- 仍返回 `status="success"`；
- 方法名被标记为 `resnet50_knn_ca2_full`。

此外，代码没有正常样本特征库，也没有 KNN。它只是使用 ImageNet ResNet50 提取特征，再将特征与自身比较。

这是明确的逻辑错误，不是模型精度问题。

## 6.2 SLF-YOLO 可能加载通用 COCO 模型

若没有领域权重，代码优先加载通用 `yolov8n.pt`。该模型默认识别人、车和常见物体，不是航空发动机金属表面缺陷模型。

但代码可能仍将其输出标记为领域缺陷检测结果和完整 YOLO 方法。

替代 Sobel 基线也没有目标级定位能力：它根据边缘比例决定是否存在缺陷，并在异常时返回整张图作为一个 bounding box。

本地测试中随机噪声图像被判为表面缺陷并返回整图框：

```text
status=success
method=edge_gradient_baseline
defects_found=1
bbox=[0,0,256,256]
confidence≈0.50
```

## 6.3 SAM Adapter 不是真正的裂纹适配器

当前路径主要是：

- 加载标准 SAM；
- 用图像中心点作为正提示；
- 取预测 mask；
- 按 mask 面积比例判断“裂纹”。

没有看到：

- 裂纹检测器提供的提示点/框；
- 裂纹专用 LoRA/Adapter 权重；
- 细长形态约束；
- 骨架连续性；
- 宽度、分支和方向判定；
- 对孔探反光、边缘、涂层纹理的误检抑制。

回退路径把 Canny 边缘当作裂纹，任何结构边缘都可能成为“裂纹”。

## 6.4 EGCIENet 和 TS-SAM 名称与实现不一致

相关适配器并未真正加载对应论文模型，部分路径使用通用 YOLOv8 segmentation 或复用其他适配器。这会产生严重的模型身份问题：

- Manifest 中的验证指标属于论文模型；
- 实际执行的是另一个模型或启发式；
- 输出仍使用论文模型资产 ID；
- 资产验证结论无法转移到实际执行实现。

这是典型的“指标借用”风险。

---

## 7. 信号检测资产审计

## 7.1 WCamba 无权重时可使用随机 CNN

代码在未找到真实 WCamba 仓库/权重时会动态创建一个 `Simple1DCNN`，没有训练步骤，也没有加载权重，却设置：

```python
self._loaded = True
self._model_type = "1dcnn_baseline"
```

随后 softmax 输出可能被作为故障概率并返回 `success`。

如果找到了 WCamba 仓库但没有 `best_model.pth`，代码也可能使用随机初始化的仓库模型并标记为已加载。

**随机网络的概率没有统计意义，不允许以任何方式进入诊断证据。**

## 7.2 WCamba 的频谱基线缺乏轴承几何参数

回退基线使用固定频率倍数和默认转速判断故障，但轴承 BPFO、BPFI、BSF、FTF 需要：

- 滚动体数量；
- 滚动体直径；
- 节圆直径；
- 接触角；
- 轴速；
- 采样和传感器位置。

没有这些参数时，固定倍频并不能代表具体发动机轴承故障频率。

## 7.3 FaultSense LSTM 可能创建未训练自编码器

代码允许动态创建 LSTM Autoencoder 后将其标记为已加载，但没有训练权重时重构误差没有可解释性。

同时，`run()` 中没有看到可靠地调用 `_load_model()` 的路径，完整模型可能根本不会被加载，实际长期使用趋势基线。

趋势基线仅根据线性斜率和硬编码阈值判异常，没有：

- 工况归一化；
- 传感器间协变量处理；
- 健康基线；
- 分段工况；
- 置信区间；
- OOD；
- 传感器漂移和缺失处理。

---

## 8. 损伤表征资产审计

## 8.1 裂纹几何测量存在负面积路径

代码试图猜测 bbox 格式，但没有显式 Schema。对常见 `[x, y, width, height]` 输入 `[10,10,20,5]`，它把后两个值当成 `x2,y2`，计算：

- 宽度 `20-10=10`；
- 高度 `5-10=-5`；
- 面积 `-50`。

本地执行结果：

```text
status=success
length_px=10.0
width_px=-5.0
area_px2=-50.0
```

程序没有拒绝负宽度、负面积，也没有输出校验。

## 8.2 没有真正计算裂纹长度

虽然默认参数包含 `skeletonize_method="zhang_suen"`，实现没有进行骨架化。所谓长度是 bbox 的长边，面积是 bbox 矩形面积。

真实裂纹几何应至少包括：

- mask 验证；
- skeleton；
- 主路径长度；
- 最大/平均宽度；
- 分支数；
- 曲率；
- 方向；
- 端点；
- 标定误差；
- 重复测量不确定性。

## 8.3 Mask URI 未被读取

若传入 `segmentation_mask_uri` 字符串，代码没有可靠加载文件，可能使用空白 mask 继续运行。这使 Artifact 引用和真实数据处理脱节。

## 8.4 损伤类型分类器只是关键词规则

DamageTypeClassifier 根据文本中是否出现 `crack`、`TBC`、`FOD` 等词分类。

它读取 `score_semantics`，但没有根据概率、异常分数、相似度或规则分数的不同语义处理。任何数值 `score > 0.9` 都可能把 `suspected` 提升为 `inferred`。

例如异常分数 0.95 并不等价于“裂纹类别概率 95%”。

## 8.5 严重度规则存在单位错误和过度泛化

SeverityRater 使用固定阈值：

- 裂纹 0.5/2/5 mm；
- 涂层剥落 2/10/30%；
- FOD 深度固定阈值。

这些规则没有绑定：

- 发动机型号；
- 具体部件位置；
- 材料；
- 维修手册版本；
- 检查方式；
- 可测量性和误差。

涂层规则还把 `area_mm2` 与 `critical_area_percent` 比较，属于量纲错误。

本地测试中输入 15 mm² 涂层面积后，代码以百分比阈值逻辑判定严重度，仍返回 `success`。

---

## 9. 数据质量门审计

## 9.1 当前 DataQualityGate 不能承担门控责任

主要检查行为：

- 字典非空即认为有内容；
- `artifact_id` 和 `artifact_type` 缺失只警告；
- 列表仅检查 NaN/Inf；
- 标量数值没有范围检查；
- 单位字典中任何字符串都判通过；
- 来源只检查是否有 `producer_asset_id`；
- fitness 永远通过；
- privacy 只检查简单分类字符串。

默认参数中的 `check_*` 开关没有真正控制检查执行。

## 9.2 对抗测试结果

以下输入包含：

- `temperature=-999`；
- 非法单位；
- 缺少有效时间和标定信息。

结果却是：

```text
overall_status=pass
recommendation=proceed
status=success
```

一个 `data=[]` 的空图像 Artifact 也被判为 `pass/proceed`。

仅有 `{"foo":123}` 的对象被判为 `warn`，而不是阻断。

## 9.3 工具运行状态与质量状态冲突

即使内部 `overall_status="fail"`，外层 `AssetRunResult.status` 仍可能是 `success`。如果调用方只检查外层状态，就会继续执行。

正确语义应当是：

- 程序执行成功；
- 数据质量结论失败；
- 后续计划必须被阻断。

这三件事必须以强类型字段分别表达，并由状态机强制执行。

---

## 10. 可靠性和 RUL 审计

## 10.1 CNN-LSTM RUL 实际没有 CNN-LSTM

`CNNLSTMRULPredictor` 的实现只做：

- 在各传感器上拟合线性趋势；
- 选择斜率绝对值最大的传感器；
- 用固定最大退化率线性换算 RUL；
- 给出固定 ±15 周期区间。

代码注释称“安装 TensorFlow + 权重可获得 CNN-LSTM”，但实现中没有模型加载和推理逻辑。

因此资产 ID、Manifest 指标与实际代码不一致。

## 10.2 Paris Law 使用危险默认参数

Paris 工具默认：

- `C=1e-12`；
- `m=3.0`；
- 初始裂纹 1 mm；
- 临界裂纹 10 mm；
- 应力范围 200 MPa；
- 几何因子 1.12。

因为默认值已合并，`validate_inputs()` 即使调用者没有提供材料和载荷参数也能通过。

其他问题：

- 参数中的 `geometry_factor` 未被使用，代码硬编码 1.12；
- 导入 `py_fatigue` 后仍调用自有积分，只改变方法标签；
- 没有检查 `critical_crack > initial_crack`；
- 没有 ΔKth/Kc/短裂纹/高温蠕变适用域；
- 没有材料批次、温度、应力比和载荷谱；
- 循环数使用 `int(dN)` 每步截断。

该工具只能作为数值演示，不可作为发动机寿命评估。

## 10.3 其他可靠性资产存在“名称大于实现”问题

已观察到：

- PINN 资产明确未实现；
- ChangePoint-LSTM 实际是分段线性基线；
- pyLife 资产可能直接使用 Basquin 公式而未调用对应库；
- 概率裂纹模型使用通用默认分布，并静默限制样本数。

这些资产却有一部分在清单中被标记为 `validated` 或 `qualified`。

---

## 11. 专家知识审计

## 11.1 当前知识库不是文档检索系统

`expert_knowledge_base.py` 是一个手写的 Python 列表，共 29 条知识项：

- 4 条术语；
- 9 条机理；
- 6 条工程规则；
- 3 条标准条款；
- 7 条专家经验。

其中 24 条自标记为证据等级 A，5 条为 B，但仓库没有证据等级审批流程。

29 条均有 `source_ref`，但只有 1 条包含 `source_location`。多数没有：

- 页码；
- 表号；
- 章节号；
- 文档修订版；
- 生效日期；
- 适用航空公司/运营人；
- 文件哈希；
- 原文摘录；
- 解析过程；
- 审核签名。

## 11.2 检索不是 RAG

`OminKnowledgeSource` 使用：

- 字符串清洗；
- 中文 2–4 字滑窗；
- 手写同义词；
- 子串匹配；
- 简单加权分数。

没有：

- 文档解析；
- OCR；
- chunk；
- BM25；
- embedding；
- reranker；
- 引用定位；
- 文档版本索引；
- 权限控制；
- 检索评估。

## 11.3 发动机和材料条件被读取但没有用于过滤

代码构造了 `material_list` 和 `engine_list`，但实际过滤只有效处理部件和损伤类型。材料、发动机型号、工况、位置等没有完整硬过滤。

本地对抗测试：

```text
query = HPT crack
component = HPT_blade
damage = crack
```

分别传入：

1. `engine=CFM56-7B, material=René 125`；
2. `engine=unrelated_engine, material=wood`。

两次都返回 10 条相同的前几项知识。

这说明文档中强调的“严格适用域”没有真正落实。

## 11.4 `min_relevance` 没有应用

默认参数包含 `min_relevance=0.5`，实际结果追加条件是：

```python
if score > 0 or component or damage or ktype:
```

只要查询提供了 component 或 damage，即使相关性分数为 0，也可能加入结果。

## 11.5 排除域只检查部件

知识项定义了材料、发动机型号、涂层类型和损伤形态等 exclusions，但检索代码主要只处理 `exclusions.component`。这使“禁止跨场景使用”的设计无法兑现。

## 11.6 高风险知识缺乏可验证原文

知识表中包含诸如：

- 可见裂纹阈值；
- 立即停飞；
- 固定检查周期；
- 打磨深度和粗糙度；
- 振动/性能阈值；
- FOD 寿命降低百分比。

这些都可能直接影响维修决策，但多数只有文件名而无页码/表号和修订版。系统无法自动验证模型引用的数值是否来自正确机型和正确版本。

## 11.7 Boeing 和 MaintIE 资产是占位信息

BoeingKnowledgeNER 返回一条固定知识，其中包含 HPT 故障百分比及“Boeing SDR analysis FY2024”来源，但没有数据、查询、文件和可复现实验。

MaintIEKnowledgeSource 只返回本体类名和仓库提示，不提供真实知识检索。

---

## 12. 风险规则和维修建议审计

## 12.1 风险矩阵过小且命名不一致

RiskClassificationRules 只含少量 `(severity, component)` 映射。组件使用 `"HPT Blade"`，而系统其他位置常使用 `"HPT_blade"` 或 `"HPT Blade Stage 1"`。

当组件字符串不匹配时走默认逻辑。

本地测试中：

```text
severity=high
damage_type=crack
component=HPT_blade
```

因为 `high` 不是该规则预期的严重度枚举，结果为：

```text
risk_level=medium
requires_review=False
```

这对安全关键系统不可接受。未知组合应当返回 `unknown/needs_review`，绝不能默认降级。

## 12.2 并非所有安全决策都需要复核

规则只在 `critical/high` 时设置 `requires_review=True`。这与架构中“最终安全关键结论必须由专家签署”的原则冲突。

即使是继续服役、常规复检或低风险判断，也可能是安全关键决策。

## 12.3 复检周期是通用硬编码

InspectionIntervalRules 使用固定周期：

- critical：0；
- high：25；
- medium：100；
- low：300；
- negligible：1000。

没有考虑：

- 发动机型号；
- ATA 章节；
- 部件；
- 损伤位置和形态；
- 运营环境；
- 载荷谱；
- OEM/运营人维修方案；
- 当前检测置信度；
- 裂纹扩展分布；
- 检查能力 POD。

固定周期不应出现在通用规则中。

---

## 13. 专家复核审计

## 13.1 审核身份可以伪造

ReviewService 的 `approve()` 接受：

- `reviewer` 字符串；
- `reviewer_role` 字符串。

没有：

- 登录身份；
- 组织目录；
- 资质证书；
- 角色授权；
- 证书有效期；
- 冲突检查；
- 二次确认；
- 数字签名验证。

## 13.2 数据模型要求签名，服务却不生成签名

`ReviewDecision` 有：

- `reviewer_credentials`；
- `signature_hash`；
- `signed_at`；
- `evidence_package_id`。

ReviewService 创建批准记录时没有填写或验证关键字段。

## 13.3 不校验证据包

审核请求可以使用空 `evidence_package_id`，批准时也没有检查：

- 证据包是否存在；
- 哈希是否匹配；
- 所有必需证据是否齐全；
- 是否有冲突证据未处理；
- 计划和资产版本是否冻结；
- 审核人是否看过相同版本。

## 13.4 审核与任务状态没有事务联动

批准 Review 不会原子地：

- 锁定 DecisionDraft；
- 验证任务状态；
- 写入事件；
- 更新任务；
- 封存证据包；
- 验证签名；
- 防止重复批准。

---

## 14. 资产治理与验证声明审计

## 14.1 清单中的验证状态与实际实现不匹配

31 个资产中：

- 19 个标记 `validated`；
- 6 个标记 `qualified`；
- 3 个 `candidate`；
- 3 个 `draft`。

也就是说 25 个资产被视为可通过 PlanCompiler。

然而：

- 31 个资产的 `verification.report_uri` 全为空；
- 所有 `implementation_ref` 为空；
- 多个 reviewer 是“authors”“community”“multiple research groups”等描述，不是对当前实现版本的签署；
- 论文模型的指标被复制到 Manifest，但实际代码可能执行通用模型或启发式；
- 没有权重哈希、代码 commit、数据 split、运行环境和复现实验记录。

## 14.2 论文结果不能自动继承给适配实现

一个资产只有在以下全部一致时，才能继承论文/上游指标：

- 模型结构；
- 权重；
- 预处理；
- 类别定义；
- 阈值；
- 数据集版本；
- split；
- 评价脚本；
- 运行环境。

当前多个资产只是使用上游项目名和指标，实际代码没有这些模型权重或实现，因此不能标记为 validated。

---

## 15. API、持久化和供应链问题

## 15.1 API 依赖中的 AssetRegistry 默认是空的

`get_asset_registry()` 只创建 `AssetRegistry()`，没有调用 `register_all_official_assets()`。因此 API 的资产搜索在默认启动后可能为空，而根目录脚本则手工注册资产。

## 15.2 领域依赖没有写入项目依赖

`pyproject.toml` 仍以 `seekflow` 为项目名，核心依赖中没有完整声明：

- FastAPI/ASGI；
- NumPy/SciPy；
- OpenCV；
- scikit-learn；
- PyTorch/torchvision；
- ultralytics；
- TensorFlow；
- 专业可靠性库。

干净安装不能保证领域模块可运行。

## 15.3 存储主要是内存和本地文件原型

任务和审核主要保存在进程内字典中。重启、多进程和多副本部署会丢失或不一致。

## 15.4 API 缺少真实安全边界

观察到开发态 CORS 和缺少认证/授权。航空维修数据和批准操作至少需要：

- OIDC/OAuth2；
- RBAC/ABAC；
- 租户和机队隔离；
- 审核角色；
- 防重放；
- 请求审计；
- 数据分类；
- 加密与密钥管理。

---

## 16. 测试体系审计

主 `tests/` 目录大量覆盖 SeekFlow 的通用运行时、安全、缓存、工具和协议，但没有看到等量的 `aero_diag` 领域测试。

领域测试缺失包括：

- Agent 是否真实调用领域工具；
- Agent 是否会在数据不足时停止；
- Agent 是否引用知识源；
- 计划完整拓扑校验；
- Schema 和单位契约；
- 资产适用域；
- 模型权重缺失时 fail-closed；
- OOD；
- 几何物理不变量；
- 规则版本；
- 专家签名；
- 证据包完整性；
- 真实数据 benchmark；
- 错误案例回归；
- 模型漂移。

根目录的两个脚本没有断言，且默认 pytest 不执行，不能充当回归测试。

---

# 17. 修复总原则

修复不应从“换一个更强 LLM”开始。正确顺序是：

1. **先让系统诚实表达能力和失败；**
2. **再建立确定性执行与验证；**
3. **再接入真实模型和知识；**
4. **最后让 Agent 负责有限范围的规划与解释。**

必须贯彻：

> LLM 只能提出假设和候选计划；确定性系统决定能否执行；已验证工具产生证据；规则引擎限定决策；授权专家签署。

---

# 18. 分阶段修复路线

## Phase 0：立即阻断危险误用

### P0-1 全部领域输出标记为“研究原型”

在 API、README 和报告中明确：

- 不用于真实维修放行；
- 不生成可执行维修命令；
- 不生成确定复检周期；
- 不生成适航结论。

### P0-2 下调资产状态

将以下类别全部改为 `draft/candidate`，直至存在当前实现的验证报告：

- CA²；
- EGCIENet；
- SLF-YOLO；
- SAM Adapter；
- TS-SAM；
- WCamba；
- FaultSense；
- CNN-LSTM RUL；
- FDPP；
- PINN；
- ChangePoint-LSTM；
- 静态知识和决策规则。

`validated/qualified` 必须由机器可验证的证据生成，不能手工填写。

### P0-3 禁止“自动回退后仍 success”

移除以下模式：

```python
try_real_model()
except:
    run_heuristic()
return success
```

改为：

```python
if validated_model_unavailable:
    return AssetRunResult(
        execution_status="unavailable",
        decision_validity="invalid",
        can_influence_decision=False,
        reason_code="MODEL_BUNDLE_UNAVAILABLE",
    )
```

基线算法如需保留，必须注册为独立资产 ID，例如：

```text
detector.surface.sobel_demo_baseline
```

并固定 `status=draft`、`can_influence_decision=false`。

### P0-4 禁止未知组合默认降风险

风险规则遇到未知输入、命名不匹配或缺失适用域时必须返回：

```json
{
  "risk_level": "unknown",
  "requires_review": true,
  "decision_blocked": true,
  "reason": "NO_APPLICABLE_APPROVED_RULE"
}
```

### P0-5 禁止匿名审批和自动归档

在认证和签名实现之前：

- 关闭批准 API；
- 禁止 E2E 自动进入 APPROVED；
- 所有结果停留在 `DECISION_DRAFT` 或 `EXPERT_REVIEW`；
- 不允许以字符串传 reviewer 身份完成批准。

---

## Phase 1：建立最小真实 Agent 闭环

## 18.1 新增 DomainAgentController

建议新增：

```text
src/aero_diag/agents/controller.py
src/aero_diag/agents/tool_bindings.py
src/aero_diag/agents/planner.py
src/aero_diag/agents/verifier.py
src/aero_diag/orchestration/executor.py
src/aero_diag/orchestration/run_store.py
```

控制流程：

```text
用户任务
  ↓
Intent Parser（结构化任务）
  ↓
Planner Agent（只能搜索和描述资产）
  ↓
PlanProposal Pydantic 校验
  ↓
PlanCompiler（确定性、fail-closed）
  ↓
人工计划审批（若需要）
  ↓
PlanExecutor（逐节点执行）
  ↓
Result Validator / Evidence Builder
  ↓
Verifier Agent（只能检查冲突和缺失，不可改原始结果）
  ↓
Decision Draft
  ↓
授权专家
```

## 18.2 给 Agent 注册稳定的平台工具

不要把 31 个具体资产直接暴露给 LLM。只暴露：

```text
search_engineering_assets
get_engineering_asset_manifest
create_plan_proposal
compile_diagnostic_plan
submit_execution_plan
get_run_status
get_artifact_summary
retrieve_authoritative_knowledge
validate_evidence_package
request_expert_review
```

`run_engineering_asset` 只允许 PlanExecutor 使用，不允许 Agent 任意直调，除非在严格沙箱和策略中。

## 18.3 强制结构化 Planner 输出

定义：

```python
class PlannerOutput(BaseModel):
    hypotheses: list[Hypothesis]
    missing_information: list[MissingInformation]
    nodes: list[ProposedNode]
    evidence_requirements: list[EvidenceRequirement]
    stop_conditions: list[StopCondition]
```

使用 `output_model` 或 JSON Schema 强制解析。解析失败必须重试或终止，不能把自然语言当计划执行。

## 18.4 Agent 必须展示真实工具调用证据

每次诊断应存储：

- 模型和 Prompt 版本；
- 工具名称；
- 参数；
- 调用时间；
- 结果状态；
- 输入/输出 Artifact 哈希；
- 失败和重试；
- Agent 如何使用结果；
- 最终每条陈述引用的 evidence ID。

最终报告不接受无 evidence ID 的事实性结论。

---

## Phase 2：修复计划编译和执行引擎

## 18.5 实现真正的拓扑校验

使用 Kahn 算法：

```python
def topological_sort(nodes):
    by_id = {n.node_id: n for n in nodes}
    indegree = {nid: 0 for nid in by_id}
    outgoing = {nid: [] for nid in by_id}

    for node in nodes:
        for dep in node.depends_on:
            if dep not in by_id:
                raise PlanError(f"unknown dependency: {dep}")
            indegree[node.node_id] += 1
            outgoing[dep].append(node.node_id)

    queue = deque(n for n, d in indegree.items() if d == 0)
    ordered = []
    while queue:
        nid = queue.popleft()
        ordered.append(by_id[nid])
        for nxt in outgoing[nid]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(ordered) != len(nodes):
        raise PlanError("cycle detected")
    return ordered
```

同时检查：

- 每个非入口节点至少有依赖；
- 每个节点从入口可达；
- 每个输出被消费或声明为最终证据；
- 阶段顺序合法；
- 不能由后续阶段依赖前置阶段。

## 18.6 扩展计划摘要

摘要至少包括：

```python
canonical = {
    "task_id": task_id,
    "task_constraints_digest": ...,
    "nodes": [...],
    "required_stages": ...,
    "evidence_requirements": ...,
    "approval": ...,
    "compiler_version": ...,
    "policy_version": ...,
    "asset_manifest_digests": ...,
    "knowledge_snapshot_digest": ...,
    "rule_set_digest": ...,
}
```

使用完整 SHA-256，不应只截断为 16 个十六进制字符作为安全标识。

## 18.7 门检查必须内建且 fail-closed

禁止把门条件仅作为字符串。每个关键迁移必须有固定验证函数：

```python
GATES = {
    (RECEIVED, DATA_VALIDATION): require_input_artifacts,
    (DATA_VALIDATION, PLAN_PROPOSAL): require_quality_pass,
    ...
}
```

若关键迁移没有 checker，应启动失败，而不是放行：

```python
if transition in REQUIRED_GATES and checker is None:
    raise GateConfigurationError
```

状态迁移、证据检查和审计事件应在同一数据库事务中提交。

## 18.8 建立真实 PlanExecutor

执行器必须支持：

- 计划摘要验证；
- 节点拓扑调度；
- 输入 Artifact 解析；
- 节点级 timeout/retry；
- 幂等键；
- checkpoint；
- 失败策略；
- 节点审批；
- 取消；
- Artifact 和 Evidence 持久化；
- 运行日志；
- 环境和依赖摘要；
- 恢复执行。

---

## Phase 3：重建资产运行契约

## 18.9 拆分执行成功和工程有效性

建议结果模型：

```python
class AssetRunResult(BaseModel):
    execution_status: Literal[
        "success", "failed", "timeout", "unavailable", "cancelled"
    ]
    validity_status: Literal[
        "valid", "degraded", "invalid", "ood", "unverified"
    ]
    can_influence_decision: bool
    reason_codes: list[str]
    outputs: list[ArtifactRef]
    evidence_items: list[EvidenceItem]
    uncertainty: UncertaintyReport
    provenance: ProvenanceRecord
```

约束：

```text
can_influence_decision=True
仅当：execution=success AND validity=valid AND asset=qualified
```

## 18.10 Runner 强制执行 Manifest

运行前：

- 资产状态必须在允许集合；
- Manifest digest 必须匹配；
- 实现 bundle 必须匹配 Manifest；
- 输入 Artifact 类型和字段必须匹配；
- 单位转换明确；
- 任务部件、材料、工况必须在 applicability；
- exclusion 命中立即拒绝；
- 参数 JSON Schema 校验；
- 权重和依赖健康检查；
- 审批策略检查。

运行后：

- 输出 JSON Schema；
- 数值范围；
- 单位；
- 物理不变量；
- OOD；
- 不确定性；
- Artifact 哈希；
- 证据关联；
- 方法和模型身份。

## 18.11 模型包必须不可伪装

每个模型 bundle：

```text
bundle/
  manifest.yaml
  model.onnx|safetensors|pth
  preprocessing.json
  class_map.json
  thresholds.json
  calibration.json
  training_data_card.md
  validation_report.json
  evaluation_outputs/
  sbom.json
  signatures/
```

Manifest 中记录：

- 代码 commit；
- 权重 SHA-256；
- 数据集 digest；
- split digest；
- 评价脚本 digest；
- 环境镜像 digest；
- 模型卡；
- 审核人数字签名；
- 有效期。

没有完整 bundle 时，模型不可加载。

---

## Phase 4：逐项修复算法资产

## 18.12 CA²

必须：

- 使用真实 CA²/MMR 等领域实现；
- 保存正常特征库或训练的密度模型；
- 严格区分图像级和像素级异常分数；
- 使用验证集校准阈值；
- 输出 domain shift/OOD；
- 删除 `feat-feat`；
- 真实方法身份与资产 ID 一致。

## 18.13 SLF-YOLO

必须：

- 没有领域权重时直接 unavailable；
- 禁止自动使用 COCO `yolov8n.pt`；
- 验证 class map 是目标缺陷类别；
- 输出每类 AP、置信度校准、最小可检尺寸；
- 在孔探域重新验证，不能直接继承 NEU-DET 指标。

## 18.14 SAM 裂纹分割

必须：

- 由上游候选检测生成 prompt；
- 加载裂纹域 Adapter/LoRA；
- 细长形态后处理；
- 反光和边缘伪影抑制；
- mask 质量评分；
- 无可靠 prompt 时返回 needs_more_evidence。

## 18.15 WCamba/FaultSense

必须：

- 权重缺失即 unavailable；
- 禁止随机网络推理；
- 加载后执行权重 checksum；
- 健康检查使用固定 golden sample；
- 输入通道、采样率和工况匹配；
- 轴承频率必须使用真实几何参数；
- 概率必须校准；
- 输出 OOD 和信噪比范围。

## 18.16 裂纹几何

输入 Schema 明确：

```json
{"bbox_format":"xyxy", "bbox":[x1,y1,x2,y2]}
```

或：

```json
{"bbox_format":"xywh", "bbox":[x,y,w,h]}
```

必须验证：

- `x2>x1`、`y2>y1` 或 `w>0`、`h>0`；
- bbox 在图像范围；
- mask 非空；
- 标定对象、标定日期和误差；
- mm 输出只在标定有效时产生。

长度使用 skeleton 最长路径，面积使用 mask 像素数。

## 18.17 DataQualityGate

建议按 ArtifactType 定义 Schema：

- 图像：尺寸、位深、编码、曝光、模糊、反光、遮挡、标定；
- 时序：采样率、时钟、缺失、饱和、漂移、同步、单位；
- 载荷谱：循环定义、时间覆盖、幅值范围；
- 维修记录：机号、部件号、版本、签名；
- 材料：牌号、批次、温度和来源。

使用 `pint` 或等价单位系统执行量纲检查。非法单位必须 fail。

## 18.18 Paris/RUL

Paris：

- 删除安全关键默认值；
- 强制材料参数和来源；
- 强制几何函数；
- 强制载荷谱；
- 检查 LEFM、ΔKth、Kc、短裂纹和温度适用域；
- 输出参数分布和敏感性；
- 与可信断裂力学工具交叉验证。

RUL：

- 资产名必须反映实际算法；
- 若只有趋势基线，改名并标记 demo；
- 真正 CNN-LSTM 需加载签名权重；
- 按 C-MAPSS 子集、传感器归一化和 RUL 标签策略复现；
- 不能将 C-MAPSS 性能直接转移到真实 CFM56 部件寿命。

---

## Phase 5：重建专家知识系统

## 18.19 建立权威来源分级

建议：

- S0：OEM 当前有效 AMM/ESM/SRM/MPD、AD、SB；
- S1：监管机构批准资料和运营人批准方案；
- S2：经专家审批的内部工程规则；
- S3：同行评审论文；
- S4：案例和经验；
- S5：模型生成或未验证内容。

只有 S0–S2 可直接支持维修决策规则。S3–S4 只能支持分析和进一步检查建议。

## 18.20 文档摄取链

```text
文件登记
→ 文件哈希和权限
→ OCR/文本解析
→ 版面和表格解析
→ 页码/章节保留
→ chunk
→ 元数据
→ 人工抽样验证
→ 索引
→ 发布快照
```

每个 chunk：

```json
{
  "document_id": "...",
  "revision": "...",
  "effective_date": "...",
  "engine_models": ["..."],
  "component": "...",
  "ata": "...",
  "page": 123,
  "table": "...",
  "text": "...",
  "file_sha256": "...",
  "access_class": "..."
}
```

## 18.21 检索必须先硬过滤再排序

顺序：

1. 权限过滤；
2. 文档有效期；
3. 发动机型号；
4. 部件/ATA；
5. 材料和配置；
6. 检查方法；
7. 排除条件；
8. BM25 + embedding；
9. reranker；
10. 引用验证。

不能只靠语义相似度处理适用域。

## 18.22 建立 Citation Verifier

在最终报告前自动检查：

- 每个安全关键数值是否有引用；
- 引用页是否包含该数值；
- 引用机型是否匹配；
- 文档是否最新有效；
- 是否引用摘要而非原文；
- 是否存在冲突条款；
- 引用是否有访问权限。

引用失败则阻断决策草案。

## 18.23 规则从知识中分离

规则不能由 LLM 临时生成。应使用版本化 DSL：

```yaml
rule_id: CFM56_7B_HPT_CRACK_DISPOSITION_001
source:
  document_id: ...
  revision: ...
  page: ...
applicability:
  engine_model: CFM56-7B
  component: HPT_STAGE_1_BLADE
conditions:
  - field: damage_type
    op: eq
    value: crack
outputs:
  decision: ENGINEERING_REVIEW_REQUIRED
  continuation_allowed: false
requires_dual_review: true
```

规则编译、测试和签署后才能发布。

---

## Phase 6：专家审批和安全治理

## 18.24 认证和资质

审核人必须来自身份系统：

- immutable user ID；
- 组织；
- 角色；
- 资质编号；
- 资质范围；
- 有效期；
- 机型授权；
- 电子签名证书。

## 18.25 证据包签名

签署对象必须是 canonical evidence package digest：

```text
task digest
+ input artifact digests
+ plan digest
+ model bundle digests
+ outputs
+ knowledge snapshot
+ rule set
+ decision draft
```

审核签名绑定该 digest。任何修改都产生新版本和新签名。

## 18.26 双人复核

至少对以下情况强制双人复核：

- 继续服役；
- 复检周期延长；
- 严重/危急损伤；
- 规则冲突；
- 模型 OOD；
- 证据不足；
- 模型和专家意见不一致。

---

# 19. 关键代码修复建议

## 19.1 Runner 统一入口

禁止业务代码直接实例化实现类。所有执行必须经过同一个 Runner，并强制：

```python
def execute(request: AssetExecutionRequest) -> AssetRunResult:
    manifest = registry.resolve_signed(request.asset_ref)
    enforce_status(manifest)
    enforce_approval(manifest, request.approval)
    validate_input_contract(manifest, request.inputs)
    enforce_applicability(manifest, request.task_context)
    verify_bundle(manifest)

    raw = isolated_runner.run(...)

    validate_output_contract(manifest, raw)
    validate_invariants(manifest, raw)
    persist_artifacts(raw)
    build_evidence(raw)
    return signed_result(raw)
```

## 19.2 资产状态发布门

```python
VALIDATION_REQUIREMENTS = {
    "validated": [
        "report_uri", "report_sha256", "code_commit", "weight_sha256",
        "dataset_digest", "split_digest", "evaluation_digest",
        "reviewer_id", "signature", "valid_until"
    ]
}
```

缺少任一项时，注册中心拒绝 `validated/qualified`。

## 19.3 输出不变量

示例：

```python
assert width_px >= 0
assert area_px2 >= 0
assert 0 <= probability <= 1
assert ci_low <= estimate <= ci_high
assert critical_crack_mm > initial_crack_mm
assert sample_rate > 0
assert all_units_are_known
```

不变量失败必须把资产运行标记为 `invalid`，不能只发 warning。

## 19.4 禁止模型身份替换

```python
if requested_asset == "detector.surface.slf_yolo_metal_defect":
    if bundle.model_family != "SLF-YOLO":
        raise ModelIdentityMismatch
```

通用 YOLO 不得冒充 SLF-YOLO，ResNet 特征不得冒充 CA² KNN，线性趋势不得冒充 CNN-LSTM。

---

# 20. 测试和验收体系

## 20.1 测试层次

### 单元测试

- bbox 格式和负值；
- 单位和量纲；
- DAG 环；
- 摘要覆盖；
- 状态门；
- 规则匹配；
- engine/material hard filter。

### 契约测试

对每个资产自动从 Manifest 生成：

- 必填字段缺失；
- 错误类型；
- 错误单位；
- OOD；
- 输出 Schema；
- 数值不变量。

### Golden tests

每个真实模型 bundle 必须有固定输入、固定输出范围和校验 hash。

### 对抗测试

- 全黑/全白/噪声图；
- 图像边缘和反光；
- 随机振动；
- 常数信号；
- 错误采样率；
- 缺失通道；
- 发动机和材料不匹配；
- Prompt injection；
- Agent 伪造工具结果；
- 规则缺失。

### 闭环集成测试

必须断言：

- Agent 实际调用了指定领域工具；
- 调用参数来自计划；
- 工具失败后 Agent 不编造结果；
- 质量门失败后计划停止；
- 知识引用包含页码；
- 决策草案只引用 evidence IDs；
- 无签名不能 APPROVED；
- approved 后任何修改使签名失效。

### 领域 benchmark

分别建立：

- 孔探检测；
- 裂纹分割；
- 尺寸测量；
- 振动故障；
- 可靠性；
- 知识检索；
- 规则一致性；
- Agent 任务完成率。

## 20.2 最低发布门

系统从“研究原型”进入“受控试验”前，应满足：

1. `/runs` 非占位且支持可恢复执行；
2. 100% 关键状态迁移有真实 gate checker；
3. 所有计划环和不可达节点被拒绝；
4. 计划 digest 覆盖证据、规则、资产和审批；
5. 无资产在缺少验证报告时标记 validated/qualified；
6. 无模型在权重缺失时使用随机网络并返回 success；
7. 无通用模型冒充领域模型；
8. 所有输出经过 Schema 和不变量校验；
9. 知识检索强制机型、部件、材料和版本过滤；
10. 所有安全关键陈述有页码级引用；
11. 风险规则未知组合返回 blocked/unknown；
12. 所有最终决定需要可验证签名；
13. 领域测试在 CI 中运行；
14. 真实 benchmark 报告可复现；
15. 故障注入测试证明 fail-closed。

进入真实维修决策还需要额外的组织、法规、数据和认证工作，单纯代码测试不能替代适航和维修体系批准。

---

# 21. 建议问题清单

| ID | 优先级 | 问题 | 修复结果 |
|---|---|---|---|
| AER-001 | Blocker | `/runs` 未实现 | 完整 PlanExecutor 与 RunStore |
| AER-002 | Blocker | 领域 Agent 未接入 | DomainAgentController + 工具绑定 |
| AER-003 | Blocker | 状态门可绕过 | 内建 checker、fail-closed、事务化 |
| AER-004 | Blocker | DAG 循环可通过 | 完整拓扑排序和可达性检查 |
| AER-005 | Blocker | Plan digest 不完整 | 全计划 canonical digest |
| AER-006 | Blocker | 资产验证状态虚高 | 状态下调、签名验证报告发布门 |
| AER-007 | Blocker | CA² 分数恒零 | 删除错误实现，接真实模型/特征库 |
| AER-008 | Blocker | 通用 YOLO 冒充领域模型 | 模型身份强校验 |
| AER-009 | Blocker | 随机 CNN/LSTM 返回 success | 权重缺失即 unavailable |
| AER-010 | Blocker | 几何负面积仍 success | 明确 bbox Schema + 不变量 |
| AER-011 | Blocker | DataQuality 错误放行 | 类型化质量规则和阻断门 |
| AER-012 | Blocker | 风险未知组合默认 medium | unknown + mandatory review |
| AER-013 | Critical | 发动机/材料过滤失效 | 硬过滤和 exclusions 全字段执行 |
| AER-014 | Critical | 知识无页码/版本 | 文档摄取和 citation verifier |
| AER-015 | Critical | 硬编码复检周期 | 版本化规则 DSL、无规则则阻断 |
| AER-016 | Critical | 匿名审批 | OIDC、资质、数字签名 |
| AER-017 | Critical | API Registry 为空 | 启动时受控注册和健康检查 |
| AER-018 | Critical | 依赖未声明 | 拆包和锁定环境 |
| AER-019 | Critical | 领域测试不在 CI | 建立 `tests/aero_diag` |
| AER-020 | High | 内存存储 | PostgreSQL/对象存储/事件日志 |

---

# 22. 推荐目标目录

```text
src/aero_diag/
  agents/
    controller.py
    planner.py
    verifier.py
    tool_bindings.py
    schemas.py
  orchestration/
    compiler.py
    executor.py
    gates.py
    run_store.py
    events.py
  assets/
    registry.py
    bundle.py
    validator.py
    policy.py
  knowledge/
    ingestion.py
    index.py
    retriever.py
    citations.py
    rules_dsl.py
  reviews/
    identity.py
    credentials.py
    signatures.py
    service.py
  validation/
    schemas.py
    units.py
    invariants.py
    applicability.py
  persistence/
    db.py
    repositories.py
    object_store.py
  api/
    auth.py
    routes/
```

---

# 23. 最终判断

该项目的**架构理念明显强于当前实现**。其领域模型、Artifact、Evidence Graph、资产 Manifest、计划冻结和专家签署等方向是正确的，但实现中大量组件仍属于：

- 接口占位；
- 论文项目名适配；
- 通用模型替代；
- 启发式基线；
- 手写知识；
- 硬编码规则；
- 演示脚本。

当前最危险的不是这些组件简单，而是它们常被包装为：

- `validated/qualified`；
- `success`；
- “full model”；
- “expert knowledge”；
- “review required/approved”。

这会制造错误的系统可信度。

正确的修复方向不是继续添加更多 Agent 或更多算法名称，而是：

1. 先下调能力声明；
2. 让所有失败、退化和未验证状态可见；
3. 建立不可绕过的计划、质量、适用域和审批控制；
4. 只接入带权重、数据卡、验证报告和哈希的真实资产；
5. 将知识变成页码级、版本化、强适用域的权威证据；
6. 最后再让 Agent 在受控工具集合内完成规划、冲突分析和报告组织。

完成这些改造后，Agent 才会真正成为系统中的“受约束工程协调者”，而不是一段专家人设 Prompt；工具成功才会表示“工程结果有效”，而不仅是 Python 函数运行完毕；专家知识和签署也才能成为可信安全边界。

---

## 附录 A：本次本地对抗测试摘要

```text
1. PlanCompiler 接受独立循环：
   CYCLE_ACCEPTED 6fb6cfa4ce0f7796 ['a', 'b', 'c', 'd']

2. 修改证据和审批要求后 digest 不变：
   0190e02c5b87607e == 0190e02c5b87607e

3. 无输入 Artifact 状态门仍通过：
   GATE_BYPASS data_validation

4. 几何测量负面积仍 success：
   bbox=[10,10,20,5]
   width_px=-5.0, area_px2=-50.0

5. 数据质量异常值通过：
   temperature=-999 → overall_status=pass, recommendation=proceed

6. 空图像通过：
   data=[] → overall_status=pass

7. 完全不匹配的发动机和材料返回相同知识：
   CFM56-7B/René125 与 unrelated_engine/wood → 相同前 10 条

8. 风险规则错误降级：
   high + HPT_blade + crack → medium, requires_review=False
```

## 附录 B：重点检查文件

```text
src/seekflow/agent/agent.py
src/aero_diag/agents/roles.py
src/aero_diag/orchestration/plan.py
src/aero_diag/orchestration/state_machine.py
src/aero_diag/api/routes/runs.py
src/aero_diag/api/dependencies.py
src/aero_diag/services/task_service.py
src/aero_diag/services/review_service.py
src/aero_diag/plugins/official/asset_runner.py
src/aero_diag/plugins/official/assets_inventory.py
src/aero_diag/plugins/official/implementations/_base.py
src/aero_diag/plugins/official/implementations/detectors_vision.py
src/aero_diag/plugins/official/implementations/detectors_signal.py
src/aero_diag/plugins/official/implementations/characterizers.py
src/aero_diag/plugins/official/implementations/data_quality.py
src/aero_diag/plugins/official/implementations/rul_predictor.py
src/aero_diag/plugins/official/implementations/py_fatigue_runner.py
src/aero_diag/plugins/official/implementations/reliability_extended.py
src/aero_diag/plugins/official/implementations/expert_knowledge_base.py
src/aero_diag/plugins/official/implementations/knowledge_and_rules.py
test_agent_integration.py
test_e2e_pipeline.py
pyproject.toml
```
