"""工具绑定——将 AgentRole 的 allowed_tools 映射到真实的 SeekFlow Tool。

审计修复 (AER-002): 将抽象的"允许工具名"绑定到具体实现。
这些平台工具是 LLM 唯一能看到的接口（文档 7.4 节两级调用架构）。
"""

from __future__ import annotations

from typing import Any


class PlatformTool:
    """平台工具——LLM 可调用的稳定接口。

    31 个具体资产通过 AssetRegistry 后台调度，LLM 不直接看到它们。
    """

    def __init__(self, name: str, description: str,
                 parameters: dict | None = None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}

    def to_openai_function(self) -> dict:
        """转为 OpenAI function calling 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys()),
                },
            },
        }


# ── 平台工具定义（文档 7.4 节）──

SEARCH_ASSETS_TOOL = PlatformTool(
    name="search_engineering_assets",
    description=(
        "搜索可用的工程资产（检测算法、表征工具、可靠性模型、知识源等）。"
        "按类别、适用部件、损伤类型和关键词过滤。"
        "返回资产 ID、名称、版本、状态和简要描述。"
    ),
    parameters={
        "kind": {"type": "string", "description": "资产类别: detector/characterizer/reliability_model/knowledge_source/decision_rule"},
        "component": {"type": "string", "description": "适用部件: HPT_blade/fan_blade/compressor_blade/disk/bearing 等"},
        "damage_type": {"type": "string", "description": "损伤类型: crack/coating_spallation/erosion/corrosion/FOD 等"},
        "keyword": {"type": "string", "description": "关键词搜索"},
    },
)

INSPECT_ARTIFACT_TOOL = PlatformTool(
    name="inspect_artifact",
    description="检查数据 Artifact 的元数据、质量和摘要。不读取大文件内容。",
    parameters={
        "artifact_id": {"type": "string", "description": "Artifact ID"},
    },
)

RETRIEVE_KNOWLEDGE_TOOL = PlatformTool(
    name="retrieve_authoritative_knowledge",
    description=(
        "检索专家知识库（故障机理、工程规则、标准条款、专家经验）。"
        "自动检查跨域误用。返回知识条目及其适用域、证据等级和引用来源。"
    ),
    parameters={
        "query": {"type": "string", "description": "检索查询（中英文均可）"},
        "component": {"type": "string", "description": "适用部件"},
        "damage_type": {"type": "string", "description": "损伤类型"},
        "engine_model": {"type": "string", "description": "发动机型号（如 CFM56-7B）"},
        "knowledge_type": {"type": "string", "description": "知识类型: terminology/mechanism/engineering_rule/standard_clause/expert_experience"},
    },
)

PROPOSE_PLAN_TOOL = PlatformTool(
    name="propose_diagnostic_plan",
    description=(
        "提出诊断执行计划。LLM 根据任务、数据和故障假设，"
        "提出候选节点（资产调用序列）、证据需求和停止条件。"
        "输出为结构化 PlannerOutput。"
        "计划由 PlanCompiler 进行静态校验后才可执行。"
    ),
    parameters={
        "task_id": {"type": "string", "description": "诊断任务 ID"},
        "hypotheses_json": {"type": "string", "description": "故障假设 JSON 数组"},
        "proposed_nodes_json": {"type": "string", "description": "候选节点 JSON 数组"},
        "evidence_requirements_json": {"type": "string", "description": "证据需求 JSON 数组"},
    },
)

VALIDATE_EVIDENCE_TOOL = PlatformTool(
    name="validate_evidence_package",
    description="验证证据包的完整性和一致性。检查冲突、缺失证据和引用完整性。",
    parameters={
        "evidence_package_id": {"type": "string", "description": "证据包 ID"},
    },
)

REQUEST_REVIEW_TOOL = PlatformTool(
    name="request_expert_review",
    description="请求授权专家复核诊断结论。安全关键结论必须经过此步骤。",
    parameters={
        "draft_id": {"type": "string", "description": "决策草案 ID"},
        "review_type": {"type": "string", "description": "复核类型: approval/rejection/need_more_evidence"},
        "comments": {"type": "string", "description": "复核说明"},
    },
)


# ── 角色→工具绑定映射 ──

ROLE_TOOL_BINDINGS: dict[str, list[PlatformTool]] = {
    "planner": [SEARCH_ASSETS_TOOL, INSPECT_ARTIFACT_TOOL, RETRIEVE_KNOWLEDGE_TOOL, PROPOSE_PLAN_TOOL],
    "data_quality": [INSPECT_ARTIFACT_TOOL, RETRIEVE_KNOWLEDGE_TOOL],
    "detection": [SEARCH_ASSETS_TOOL, INSPECT_ARTIFACT_TOOL, RETRIEVE_KNOWLEDGE_TOOL],
    "characterization": [SEARCH_ASSETS_TOOL, RETRIEVE_KNOWLEDGE_TOOL],
    "reliability": [SEARCH_ASSETS_TOOL, RETRIEVE_KNOWLEDGE_TOOL],
    "decision": [RETRIEVE_KNOWLEDGE_TOOL, VALIDATE_EVIDENCE_TOOL, REQUEST_REVIEW_TOOL],
    "monitor": [INSPECT_ARTIFACT_TOOL, RETRIEVE_KNOWLEDGE_TOOL],
}


def get_tools_for_role(role_id: str) -> list[PlatformTool]:
    """获取指定角色的平台工具列表。"""
    return ROLE_TOOL_BINDINGS.get(role_id, [])


def get_all_platform_tools() -> list[PlatformTool]:
    """获取所有平台工具（去重）。"""
    seen = set()
    tools = []
    for tool_list in ROLE_TOOL_BINDINGS.values():
        for tool in tool_list:
            if tool.name not in seen:
                seen.add(tool.name)
                tools.append(tool)
    return tools
