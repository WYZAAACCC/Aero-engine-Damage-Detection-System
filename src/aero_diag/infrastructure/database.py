"""数据库管理——SQLite/PostgreSQL 会话和初始化。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatabaseSession:
    """数据库会话抽象——原型使用内存字典，生产使用 SQLAlchemy。

    P0 阶段先用内存存储验证核心逻辑，P1 切换到 SQLite/PostgreSQL。
    """

    tables: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        self.tables = {
            "tasks": {},
            "plans": {},
            "runs": {},
            "reviews": {},
            "events": {},
        }

    def insert(self, table: str, item_id: str, data: dict[str, Any]) -> None:
        """插入一条记录。"""
        if table not in self.tables:
            self.tables[table] = {}
        self.tables[table][item_id] = data

    def get(self, table: str, item_id: str) -> dict[str, Any] | None:
        """获取一条记录。"""
        return self.tables.get(table, {}).get(item_id)

    def list(self, table: str) -> list[dict[str, Any]]:
        """列出表的所有记录。"""
        return list(self.tables.get(table, {}).values())

    def delete(self, table: str, item_id: str) -> bool:
        """删除一条记录。"""
        if table in self.tables and item_id in self.tables[table]:
            del self.tables[table][item_id]
            return True
        return False


# 全局数据库会话
db_session = DatabaseSession()
