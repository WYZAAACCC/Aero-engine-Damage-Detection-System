"""DomainAgentController — 航空领域 Agent 工厂与生命周期管理。

审计修复 (AER-002): 替代原来的"只有角色元数据"状态。
将 AgentRole 实例化为真正的 DeepSeekAgent，绑定领域工具，强制执行策略。

控制流程（文档 18.1 节）:
    用户任务 → Intent Parser → Planner Agent → PlanProposal 校验
    → PlanCompiler (fail-closed) → PlanExecutor → Result Validator
    → Verifier Agent → Decision Draft → 授权专家
"""

from __future__ import annotations

import logging
from typing import Any

from aero_diag.agents.roles import AgentRole, ALL_ROLES, get_role
from aero_diag.agents.tool_bindings import get_tools_for_role, PlatformTool

logger = logging.getLogger("aero_diag.agents.controller")


class DomainAgentController:
    """领域 Agent 控制器——管理所有航空诊断 Agent 的实例化和执行。

    用法:
        controller = DomainAgentController(
            api_key="sk-...",
            asset_registry=registry,
            asset_runner=runner,
        )
        planner = controller.create_agent("planner")
        result = controller.run_planning(task_description="...")
    """

    def __init__(
        self,
        api_key: str,
        *,
        asset_registry: Any = None,
        asset_runner: Any = None,
        knowledge_source: Any = None,
        model: str = "deepseek-v4-pro",
        thinking: bool = True,
        mode: str = "stable",
        max_steps: int = 20,
    ):
        self._api_key = api_key
        self._asset_registry = asset_registry
        self._asset_runner = asset_runner
        self._knowledge_source = knowledge_source
        self._model = model
        self._thinking = thinking
        self._mode = mode
        self._max_steps = max_steps
        self._agents: dict[str, Any] = {}  # role_id → DeepSeekAgent

        # ── 平台工具实现（回调）──
        self._tool_handlers: dict[str, callable] = {}

    def register_tool_handler(self, tool_name: str, handler: callable) -> None:
        """注册平台工具的实际执行回调。

        这些回调在 Agent 调用工具时触发，负责：
        - 调用 AssetRunner 执行具体资产
        - 查询 AssetRegistry 返回结果
        - 检索知识库
        """
        self._tool_handlers[tool_name] = handler

    def create_agent(self, role_id: str) -> Any | None:
        """从角色定义创建真实的 DeepSeekAgent 实例。

        Args:
            role_id: "planner" / "detection" / "characterization" 等

        Returns:
            DeepSeekAgent 实例，已绑定对应角色工具
        """
        if role_id in self._agents:
            return self._agents[role_id]

        try:
            role = get_role(role_id)
        except ValueError as e:
            logger.error(f"Unknown role: {role_id} — {e}")
            return None

        # 获取该角色的平台工具
        platform_tools = get_tools_for_role(role_id)

        # 构建 SeekFlow 兼容的工具函数列表
        seekflow_tools = []
        for pt in platform_tools:
            handler = self._tool_handlers.get(pt.name)
            if handler:
                seekflow_tools.append(_wrap_platform_tool(pt, handler))
            else:
                logger.warning(
                    f"Tool '{pt.name}' has no registered handler. "
                    f"Agent '{role_id}' will not be able to use it."
                )

        # 创建 DeepSeekAgent
        try:
            from seekflow import DeepSeekAgent

            agent = DeepSeekAgent(
                role=role.name,
                goal=role.description,
                backstory=role.system_prompt,
                api_key=self._api_key,
                model=self._model,
                thinking=self._thinking,
                mode=self._mode,
                max_steps=self._max_steps,
            )

            # 注册领域工具
            if seekflow_tools:
                for tool in seekflow_tools:
                    agent.register_tool(tool)

            self._agents[role_id] = agent
            logger.info(
                f"Agent '{role_id}' created with {len(seekflow_tools)} tools. "
                f"Forbidden: {role.forbidden_actions}"
            )
            return agent

        except ImportError:
            logger.error(
                "seekflow.DeepSeekAgent not available. "
                "Agent creation requires seekflow package installed."
            )
            return None

    def create_all_agents(self) -> dict[str, Any]:
        """创建所有 7 个角色对应的 Agent 实例。"""
        agents = {}
        for role_id in ALL_ROLES:
            agent = self.create_agent(role_id)
            if agent:
                agents[role_id] = agent
        return agents

    def get_agent(self, role_id: str) -> Any | None:
        """获取已创建的 Agent 实例。"""
        return self._agents.get(role_id)

    def run_planning(self, task_description: str,
                     task_context: dict | None = None) -> dict[str, Any]:
        """运行 Planner Agent 生成诊断计划。

        Args:
            task_description: 诊断任务的自然语言描述
            task_context: 补充上下文（部件/工况/已有数据等）

        Returns:
            {"agent_result": AgentResult, "plan_parsed": PlannerOutput|None, "error": str|None}
        """
        planner = self.get_agent("planner")
        if planner is None:
            planner = self.create_agent("planner")
        if planner is None:
            return {"error": "Planner agent not available — seekflow not installed"}

        context_str = ""
        if task_context:
            context_str = (
                f"\n\n【任务上下文】\n"
                f"- 发动机型号: {task_context.get('engine_type', '未指定')}\n"
                f"- 目标部件: {task_context.get('component', '未指定')}\n"
                f"- 运行工况: {task_context.get('operating_mode', '未指定')}\n"
                f"- 可用数据: {task_context.get('available_data', '未指定')}\n"
            )

        prompt = (
            f"{task_description}\n\n"
            f"请使用你的工具完成以下步骤：\n"
            f"1. 用 search_engineering_assets 查找适用的检测算法和表征工具\n"
            f"2. 用 retrieve_authoritative_knowledge 查询相关的故障机理和工程规则\n"
            f"3. 用 propose_diagnostic_plan 生成结构化的诊断计划\n"
            f"{context_str}\n"
            f"注意：你必须实际调用工具获取信息，不能编造资产 ID 或知识内容。"
        )

        try:
            result = planner.run(prompt, max_cost=0.30)
            return {
                "agent_result": result,
                "final_output": result.final_output,
                "tool_calls": result.tool_calls,
                "cost_cny": result.cost,
                "tokens": result.tokens,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Planner agent run failed: {e}")
            return {"error": str(e)}

    def run_diagnosis(self, task_description: str,
                      task_context: dict | None = None) -> dict[str, Any]:
        """运行完整的诊断 Agent 链。

        由 Planner → Detection → Characterization →
        Reliability → Decision 角色依次协作。
        当前为顺序调用模式，生产环境应使用工作流编排。
        """
        import json

        results = {}

        # 1. Planner: 生成计划
        plan_result = self.run_planning(task_description, task_context)
        results["planning"] = plan_result

        if plan_result.get("error"):
            return {"status": "planning_failed", "results": results}

        # 2. Decision: 综合分析
        decision_agent = self.get_agent("decision")
        if decision_agent is None:
            decision_agent = self.create_agent("decision")

        if decision_agent:
            try:
                context_info = json.dumps(task_context or {}, ensure_ascii=False)
                synthesis_prompt = (
                    f"基于以下诊断结果，生成最终诊断报告:\n\n"
                    f"【任务】{task_description}\n"
                    f"【上下文】{context_info}\n"
                    f"【规划结果】{plan_result.get('final_output', '')[:500]}\n\n"
                    f"请生成结构化的诊断结论。"
                    f"注意：你必须标记 requires_review=True，"
                    f"最终安全关键结论必须由授权专家签署。"
                )
                decision_result = decision_agent.run(synthesis_prompt, max_cost=0.20)
                results["decision"] = {
                    "final_output": decision_result.final_output,
                    "cost_cny": decision_result.cost,
                }
            except Exception as e:
                results["decision"] = {"error": str(e)}
        else:
            results["decision"] = {"error": "Decision agent not available"}

        return {"status": "completed", "results": results}

    # ── 默认工具处理器注册 ──

    def register_default_handlers(self) -> None:
        """注册平台工具的默认处理器（需要 AssetRegistry + AssetRunner）。

        在 Agent 调用平台工具时触发这些回调。
        """
        if self._asset_registry is not None:
            def search_assets(kind=None, component=None, damage_type=None, keyword=None, **kwargs):
                from aero_diag.assets.manifest import AssetKind
                kind_enum = AssetKind(kind) if kind else None
                manifests = self._asset_registry.search(
                    kind=kind_enum, component=component, keyword=keyword,
                )
                return {
                    "assets_found": len(manifests),
                    "assets": [
                        {
                            "asset_id": m.asset_id,
                            "name": m.name,
                            "version": m.version,
                            "kind": m.asset_kind.value,
                            "status": m.status.value,
                            "description": m.description[:200],
                        }
                        for m in manifests[:15]
                    ],
                }

            self.register_tool_handler("search_engineering_assets", search_assets)

        if self._asset_runner is not None:
            # 不直接暴露 run_engineering_asset 给 LLM
            # Planner 只能搜索和描述资产，不能直接执行
            pass

        if self._knowledge_source is not None:
            def retrieve_knowledge(query, component=None, damage_type=None,
                                   engine_model=None, knowledge_type=None, **kwargs):
                result = self._knowledge_source.run(
                    inputs=[{"query": query}],
                    parameters={
                        "component": component or "",
                        "damage": damage_type or "",
                        "engine_model": engine_model or "",
                        "knowledge_type": knowledge_type or "",
                    },
                    context={},
                )
                output = result.structured_output
                return {
                    "results_found": output.get("results_found", 0),
                    "total_indexed": output.get("total_indexed", 0),
                    "top_items": [
                        {
                            "title": item.get("title", ""),
                            "type": item.get("type", ""),
                            "evidence_level": item.get("evidence_level", ""),
                            "relevance_score": item.get("relevance_score", 0),
                            "content": item.get("content", "")[:300],
                            "applicability_warning": item.get("applicability_warning"),
                        }
                        for item in output.get("knowledge_items", [])[:5]
                    ],
                    "cross_domain_warnings": output.get("cross_domain_warnings", []),
                }

            self.register_tool_handler("retrieve_authoritative_knowledge", retrieve_knowledge)


def _wrap_platform_tool(pt: PlatformTool, handler: callable) -> callable:
    """将 PlatformTool + handler 包装为 SeekFlow 兼容的工具函数。"""
    def tool_func(**kwargs):
        return handler(**kwargs)

    tool_func.__name__ = pt.name
    tool_func.__doc__ = pt.description
    # SeekFlow @tool 装饰器需要 __seekflow_tool__ 属性
    tool_func.__seekflow_tool__ = {
        "name": pt.name,
        "description": pt.description,
        "parameters": pt.to_openai_function()["function"]["parameters"],
    }
    return tool_func
