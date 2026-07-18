"""资产实现——适配 EngineeringAsset 协议的载体。

本模块负责：
- 调用资产的 validate_inputs() 验证输入是否满足资产要求
- 调用资产的 run() 执行资产并返回结构化结果
- 确保资产不超出适用域限制（applicability 检查）
- 记录运行溯源（资产版本、参数快照、时间戳）
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AssetRunResult:
    """资产运行结果——与 contracts/assets.py 的 AssetRunResult 兼容。"""
    status: str = "success"  # success / partial / failed / needs_review
    structured_output: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    elapsed_ms: int = 0
    output_artifacts: list = field(default_factory=list)
    evidence_items: list = field(default_factory=list)


class ImplementationBase(ABC):
    """资产实现基类——所有实现都继承此基类。

    子类必须:
    - 设置 asset_id 类属性
    - 实现 run() 方法
    - 可选重写 validate_inputs() 和 health_check()
    """

    asset_id: str = ""

    @abstractmethod
    def run(self, inputs: list[dict], parameters: dict, context: dict) -> AssetRunResult:
        """执行资产。"""
        ...

    def validate_inputs(self, inputs: list[dict], parameters: dict, context: dict) -> dict:
        """验证输入——默认总是通过，子类可覆盖。"""
        return {"ok": True, "issues": []}

    def health_check(self) -> dict:
        """健康检查——默认总是健康，子类可覆盖。"""
        return {"healthy": True, "details": {}}

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
