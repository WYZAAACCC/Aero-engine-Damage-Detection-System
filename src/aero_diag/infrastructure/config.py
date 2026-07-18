"""配置管理——系统全局配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class AeroDiagConfig:
    """aero_diag 系统配置。

    支持环境变量覆盖，适用于开发、实验室、内网和规模化部署。
    """

    # 数据库
    database_url: str = field(default_factory=lambda: os.environ.get(
        "AERO_DIAG_DB_URL", "sqlite:///data/aero_diag.db",
    ))

    # 对象存储
    artifact_root: str = field(default_factory=lambda: os.environ.get(
        "AERO_DIAG_ARTIFACT_ROOT", "./data/artifacts",
    ))

    # 服务
    host: str = field(default_factory=lambda: os.environ.get("AERO_DIAG_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("AERO_DIAG_PORT", "8000")))

    # 日志
    log_level: str = field(default_factory=lambda: os.environ.get("AERO_DIAG_LOG_LEVEL", "INFO"))

    # 安全
    api_key_required: bool = False
    max_task_size_bytes: int = 10_000_000

    # 执行
    default_timeout_s: float = 300.0
    max_parallel_tasks: int = 5

    # 部署形态
    deployment_mode: str = "development"  # "development" | "laboratory" | "air_gapped" | "production"


# 全局配置实例
config = AeroDiagConfig()
