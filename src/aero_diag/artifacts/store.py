"""数据管理——ArtifactStore 提供原始/衍生数据保存、哈希、格式转换和 lineage 管理。

遵循文档第 6 节设计：二进制/大数据保存在对象存储中，
消息中只传 URI、哈希、Schema 和摘要。
原型使用本地文件系统 + JSON 索引，生产可切换 MinIO/S3。
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aero_diag.domain.artifacts import ArtifactEnvelope, ArtifactRef, ArtifactType


class ArtifactStore:
    """数据管理层——原型使用本地目录。

    生产环境替换为 MinIO/S3 兼容对象存储。
    """

    def __init__(self, root_dir: str | Path = "./data/artifacts"):
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._index_path = self._root / "_index.json"
        self._index: dict[str, dict] = self._load_index()

    # ── 存储 ────────────────────────────────────────────────────────

    def put(
        self,
        content: bytes,
        artifact_type: ArtifactType,
        *,
        file_name: str | None = None,
        producer_asset_id: str | None = None,
        producer_version: str | None = None,
        run_id: str = "",
        parent_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactEnvelope:
        """存储原始数据并返回 ArtifactEnvelope。

        Args:
            content: 原始字节数据
            artifact_type: 数据类型
            file_name: 文件名（用于扩展名推断）
            producer_asset_id: 产生者资产 ID
            producer_version: 资产版本
            run_id: 运行 ID
            parent_ids: 父 Artifact ID 列表
            metadata: 额外元数据

        Returns:
            ArtifactEnvelope 引用
        """
        artifact_id = uuid.uuid4().hex[:16]
        sha = hashlib.sha256(content).hexdigest()

        # 确定存储路径
        suffix = Path(file_name).suffix if file_name else ".bin"
        rel_path = f"{artifact_type.value}/{artifact_id[:4]}/{artifact_id}{suffix}"
        store_path = self._root / rel_path
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_bytes(content)

        envelope = ArtifactEnvelope(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            uri=str(store_path),
            sha256=sha,
            producer_asset_id=producer_asset_id,
            producer_version=producer_version,
            run_id=run_id,
            parent_artifact_ids=parent_ids or [],
        )
        if metadata:
            envelope.metadata.update(metadata)

        self._index[artifact_id] = envelope.model_dump(mode="json")
        self._save_index()
        return envelope

    def put_structured(
        self,
        payload: dict[str, Any],
        artifact_type: ArtifactType,
        *,
        producer_asset_id: str | None = None,
        producer_version: str | None = None,
        run_id: str = "",
        parent_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactEnvelope:
        """存储结构化数据（不携带二进制）。"""
        artifact_id = uuid.uuid4().hex[:16]
        content_str = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        sha = hashlib.sha256(content_str.encode("utf-8")).hexdigest()

        envelope = ArtifactEnvelope(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            sha256=sha,
            producer_asset_id=producer_asset_id,
            producer_version=producer_version,
            run_id=run_id,
            parent_artifact_ids=parent_ids or [],
            payload=payload,
        )
        if metadata:
            envelope.metadata.update(metadata)

        self._index[artifact_id] = envelope.model_dump(mode="json")
        self._save_index()
        return envelope

    # ── 读取 ────────────────────────────────────────────────────────

    def get(self, artifact_id: str) -> ArtifactEnvelope | None:
        """根据 artifact_id 获取 ArtifactEnvelope。"""
        data = self._index.get(artifact_id)
        if data is None:
            return None
        return ArtifactEnvelope.model_validate(data)

    def get_ref(self, artifact_id: str) -> ArtifactRef | None:
        """获取轻量引用。"""
        envelope = self.get(artifact_id)
        if envelope is None:
            return None
        return envelope.to_ref()

    def get_content(self, artifact_id: str) -> bytes | None:
        """读取二进制数据内容。"""
        data = self._index.get(artifact_id)
        if data is None or not data.get("uri"):
            return None
        path = Path(data["uri"])
        if path.exists():
            return path.read_bytes()
        return None

    # ── 派生 ────────────────────────────────────────────────────────

    def derive(
        self,
        parent_id: str,
        content: bytes,
        artifact_type: ArtifactType,
        *,
        file_name: str | None = None,
        producer_asset_id: str | None = None,
        producer_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactEnvelope:
        """从已有 Artifact 派生新数据（保留 lineage）。"""
        parent = self.get(parent_id)
        parent_ids = (parent.parent_artifact_ids or []) + [parent_id] if parent else [parent_id]
        return self.put(
            content,
            artifact_type,
            file_name=file_name,
            producer_asset_id=producer_asset_id,
            producer_version=producer_version,
            parent_ids=parent_ids,
            metadata=metadata,
        )

    # ── Lineage ─────────────────────────────────────────────────────

    def lineage(self, artifact_id: str) -> list[ArtifactRef]:
        """获取 Artifact 的完整 derivation 链。"""
        envelope = self.get(artifact_id)
        if envelope is None:
            return []
        chain = [envelope.to_ref()]
        for pid in envelope.parent_artifact_ids:
            chain.extend(self.lineage(pid))
        return chain

    def list_by_type(self, artifact_type: ArtifactType) -> list[ArtifactRef]:
        """列出指定类型的所有 Artifact。"""
        refs: list[ArtifactRef] = []
        for aid, data in self._index.items():
            if data.get("artifact_type") == artifact_type.value:
                refs.append(ArtifactRef(
                    artifact_id=aid,
                    artifact_type=artifact_type,
                    uri=data.get("uri"),
                    sha256=data.get("sha256", ""),
                ))
        return refs

    # ── 内部 ────────────────────────────────────────────────────────

    def _load_index(self) -> dict[str, dict]:
        if self._index_path.exists():
            return json.loads(self._index_path.read_text("utf-8"))
        return {}

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2, default=str),
            "utf-8",
        )
