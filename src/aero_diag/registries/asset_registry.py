"""资产注册中心——统一管理所有工程资产的注册、检索和版本解析。

遵循文档第 7 节设计：支持 namespace:name@version 键格式，
提供按类别、适用域、状态和版本的检索能力。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aero_diag.assets.manifest import AssetKind, AssetStatus, EngineeringAssetManifest


@dataclass
class AssetEntry:
    """资产记录——Manifest + 注册元数据。"""
    manifest: EngineeringAssetManifest
    registered_at: str = ""
    registered_by: str = ""
    aliases: list[str] = field(default_factory=list)
    # runtime 引用
    implementation: Any = None  # EngineeringAsset | callable | None


class AssetNotFoundError(KeyError):
    """资产未找到。"""
    pass


class AssetStatusError(ValueError):
    """资产状态不允许执行。"""
    pass


class AssetRegistry:
    """工程资产注册中心。

    支持三种键格式：
    - asset_id: "detector.vibration.order_tracking"
    - asset_id@version: "detector.vibration.order_tracking@1.2.0"
    - asset_id@latest: "detector.vibration.order_tracking@latest"

    注册时键为 asset_id，版本在 Manifest 中管理。
    """

    def __init__(self) -> None:
        self._entries: dict[str, list[AssetEntry]] = {}  # asset_id -> [versions]

    def register(
        self,
        manifest: EngineeringAssetManifest,
        implementation: Any = None,
        aliases: list[str] | None = None,
    ) -> AssetEntry:
        """注册一个工程资产。

        同 asset_id 的多版本按注册顺序追加，resolve 默认返回最新版本。
        """
        key = manifest.asset_id
        if not key:
            raise ValueError("manifest.asset_id must not be empty")

        if manifest.digest == "":
            manifest.compute_digest()

        entry = AssetEntry(
            manifest=manifest,
            registered_at="",
            registered_by="",
            aliases=aliases or [],
            implementation=implementation,
        )

        if key not in self._entries:
            self._entries[key] = []
        self._entries[key].append(entry)

        # 同时用别名索引
        for alias in entry.aliases:
            if alias not in self._entries:
                self._entries[alias] = []
            self._entries[alias].append(entry)

        return entry

    def resolve(
        self,
        asset_query: str,
        version_constraint: str | None = None,
    ) -> AssetEntry:
        """解析资产查询，返回匹配的 AssetEntry。

        Args:
            asset_query: asset_id 或 "asset_id@version"
            version_constraint: 显式版本约束（优先于 query 中的版本）

        Returns:
            匹配的 AssetEntry

        Raises:
            AssetNotFoundError: 资产未找到
        """
        # 解析 asset_id 和版本
        if "@" in asset_query:
            asset_id, query_version = asset_query.split("@", 1)
        else:
            asset_id = asset_query
            query_version = None

        effective_version = version_constraint or query_version or "latest"

        if asset_id not in self._entries:
            raise AssetNotFoundError(f"Asset not found: {asset_id}")

        entries = self._entries[asset_id]
        if not entries:
            raise AssetNotFoundError(f"No versions registered for: {asset_id}")

        if effective_version == "latest":
            return entries[-1]

        for entry in entries:
            if entry.manifest.version == effective_version:
                return entry

        available = [e.manifest.version for e in entries]
        raise AssetNotFoundError(
            f"Version {effective_version} not found for {asset_id}. "
            f"Available: {available}"
        )

    def search(
        self,
        kind: AssetKind | None = None,
        component: str | None = None,
        status: AssetStatus | None = None,
        keyword: str | None = None,
    ) -> list[EngineeringAssetManifest]:
        """按类别、适用部件、状态和关键词检索资产。

        Returns:
            匹配的 Manifest 列表
        """
        results: list[EngineeringAssetManifest] = []
        seen: set[str] = set()

        for entries in self._entries.values():
            for entry in entries:
                m = entry.manifest
                if m.asset_id in seen:
                    continue

                if kind is not None and m.asset_kind != kind:
                    continue
                if status is not None and m.status != status:
                    continue
                if component is not None:
                    if component not in m.applicability.components:
                        continue
                if keyword is not None:
                    keyword_lower = keyword.lower()
                    if not (
                        keyword_lower in m.name.lower()
                        or keyword_lower in m.description.lower()
                    ):
                        continue

                seen.add(m.asset_id)
                results.append(m)

        return results

    def list_by_kind(self, kind: AssetKind) -> list[EngineeringAssetManifest]:
        """列出指定类别的所有资产。"""
        return self.search(kind=kind)

    def get_versions(self, asset_id: str) -> list[str]:
        """获取资产的所有已注册版本。"""
        if asset_id not in self._entries:
            return []
        return [e.manifest.version for e in self._entries[asset_id]]

    def deprecate(self, asset_id: str) -> None:
        """将资产最新版本标记为 deprecated。"""
        if asset_id not in self._entries:
            raise AssetNotFoundError(f"Asset not found: {asset_id}")
        latest = self._entries[asset_id][-1]
        latest.manifest.status = AssetStatus.DEPRECATED

    def count(self) -> int:
        """已注册的资产数量（所有版本）。"""
        return sum(len(entries) for entries in self._entries.values())
