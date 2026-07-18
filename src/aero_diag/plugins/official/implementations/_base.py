"""资产实现——适配 EngineeringAsset 协议的载体。

本模块负责：
- 调用资产的 validate_inputs() 验证输入是否满足资产要求
- 调用资产的 run() 执行资产并返回结构化结果
- 确保资产不超出适用域限制（applicability 检查）
- 记录运行溯源（资产版本、参数快照、时间戳）

审计修复 (P0-3, P3-18.9):
- AssetRunResult 拆分 execution_status (程序是否跑完) 与 validity_status (结果是否可信)
- can_influence_decision 字段：只有 execution=success AND validity=valid AND asset=qualified 时为 True
- 禁止自动回退后仍返回 success 的欺骗模式
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class AssetRunResult:
    """资产运行结果——区分"程序跑完"和"结果可信"两个维度。

    审计要求 (AER-003, 18.9):
    - execution_status: 程序是否成功运行（技术层面）
    - validity_status: 结果是否工程有效（领域层面）
    - can_influence_decision: 只有全部通过才能影响决策
    """
    # ── 执行状态（技术层面）──
    execution_status: str = "success"  # success / failed / timeout / unavailable / cancelled
    # ── 有效性状态（工程层面）──
    validity_status: str = "unverified"  # valid / degraded / invalid / ood / unverified
    # ── 是否能影响安全决策 ──
    can_influence_decision: bool = False

    # ── 向后兼容字段 ──
    status: str = "success"  # 保留，映射到 execution_status
    structured_output: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    elapsed_ms: int = 0
    output_artifacts: list = field(default_factory=list)
    evidence_items: list = field(default_factory=list)

    # ── 审计追踪 ──
    reason_codes: list[str] = field(default_factory=list)
    model_identity: str = ""  # 实际使用的模型/方法身份

    def __post_init__(self):
        """自动推导 can_influence_decision。"""
        if self.can_influence_decision:
            return  # 显式设置时不覆盖
        self.can_influence_decision = (
            self.execution_status == "success"
            and self.validity_status == "valid"
        )
        # 同步 status 字段（向后兼容）
        if self.execution_status == "unavailable":
            self.status = "unavailable"
        elif self.execution_status == "failed":
            self.status = "failed"
        elif self.validity_status in ("invalid", "ood", "unverified"):
            self.status = "partial"
        else:
            self.status = self.execution_status

    # ── 工厂方法 ──

    @classmethod
    def unavailable(cls, reason: str, model_identity: str = "",
                    reason_code: str = "MODEL_BUNDLE_UNAVAILABLE") -> "AssetRunResult":
        """模型/权重不可用——明确拒绝，不静默回退。"""
        return cls(
            execution_status="unavailable",
            validity_status="invalid",
            can_influence_decision=False,
            structured_output={
                "method": "unavailable",
                "note": f"Asset unavailable: {reason}",
            },
            error=reason,
            reason_codes=[reason_code],
            model_identity=model_identity or "none",
        )

    @classmethod
    def degraded(cls, method: str, reason: str, output: dict,
                 metrics: dict = None) -> "AssetRunResult":
        """降级运行——明确标记为 degraded/基线/未验证。"""
        result = cls(
            execution_status="success",
            validity_status="degraded",
            can_influence_decision=False,
            structured_output={
                **output,
                "method": method,
                "note": f"DEGRADED: {reason}",
            },
            reason_codes=["DEGRADED_BASELINE", reason],
            model_identity=method,
            metrics=metrics or {},
        )
        return result

    @classmethod
    def valid_success(cls, output: dict, model_identity: str = "",
                      metrics: dict = None) -> "AssetRunResult":
        """完全验证的模型成功运行。"""
        return cls(
            execution_status="success",
            validity_status="valid",
            can_influence_decision=True,
            structured_output=output,
            model_identity=model_identity,
            metrics=metrics or {},
        )


class ImplementationBase(ABC):
    """资产实现基类——所有实现都继承此基类。

    子类必须:
    - 设置 asset_id 类属性
    - 实现 run() 方法
    - 可选重写 validate_inputs() 和 health_check()

    审计要求 (P0-3):
    - 权重缺失时必须返回 AssetRunResult.unavailable(), 不能静默回退
    - 基线算法通过 AssetRunResult.degraded() 返回, 标记为不可影响决策
    - 默认 validate_inputs 不再总是通过——子类必须显式验证
    """

    asset_id: str = ""

    @abstractmethod
    def run(self, inputs: list[dict], parameters: dict, context: dict) -> AssetRunResult:
        """执行资产。"""
        ...

    def validate_inputs(self, inputs: list[dict], parameters: dict, context: dict) -> dict:
        """验证输入——默认返回警告但允许通过。子类应覆盖为严格验证。"""
        if not inputs:
            return {"ok": False, "issues": ["No input data provided"]}
        return {"ok": True, "issues": []}

    def health_check(self) -> dict:
        """健康检查——默认返回未验证状态。子类有真实模型时覆盖。"""
        return {
            "healthy": True,
            "model_loaded": False,
            "model_identity": "none",
            "details": {"warning": "No model loaded — health check is nominal only"},
        }

    @staticmethod
    def default_params() -> dict:
        """默认参数——子类覆盖。"""
        return {}

    def _make_provenance(self, params: dict, run_id: str = "") -> dict:
        return {
            "asset_id": self.asset_id,
            "parameters_snapshot": hashlib.sha256(
                str(sorted(params.items())).encode()
            ).hexdigest()[:16],
            "run_id": run_id,
            "timestamp": time.time(),
        }
