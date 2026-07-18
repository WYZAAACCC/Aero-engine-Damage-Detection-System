"""资产 API 路由——注册与检索工程资产。"""

from fastapi import APIRouter, Depends, HTTPException

from aero_diag.api.dependencies import get_asset_registry
from aero_diag.assets.manifest import AssetKind, AssetStatus, EngineeringAssetManifest
from aero_diag.registries.asset_registry import AssetRegistry

router = APIRouter(prefix="/assets", tags=["assets"])


@router.post("/register", status_code=201)
def register_asset(
    manifest_data: dict,
    registry: AssetRegistry = Depends(get_asset_registry),
) -> dict:
    """注册工程资产。"""
    try:
        manifest = EngineeringAssetManifest.model_validate(manifest_data)
        manifest.compute_digest()
        entry = registry.register(manifest)
        return {
            "asset_id": entry.manifest.asset_id,
            "version": entry.manifest.version,
            "digest": entry.manifest.digest,
            "status": entry.manifest.status.value,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/search")
def search_assets(
    kind: str = "",
    component: str = "",
    status: str = "",
    keyword: str = "",
    registry: AssetRegistry = Depends(get_asset_registry),
) -> list[dict]:
    """检索工程资产。"""
    asset_kind = AssetKind(kind) if kind else None
    asset_status = AssetStatus(status) if status else None
    results = registry.search(
        kind=asset_kind,
        component=component if component else None,
        status=asset_status,
        keyword=keyword if keyword else None,
    )
    return [m.model_dump(mode="json") for m in results]


@router.get("/{asset_id}")
def get_asset(
    asset_id: str,
    version: str = "latest",
    registry: AssetRegistry = Depends(get_asset_registry),
) -> dict:
    """获取资产详情。"""
    try:
        entry = registry.resolve(asset_id, version)
        return entry.manifest.model_dump(mode="json")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{asset_id}/versions")
def get_asset_versions(
    asset_id: str,
    registry: AssetRegistry = Depends(get_asset_registry),
) -> list[str]:
    """获取资产所有版本。"""
    versions = registry.get_versions(asset_id)
    if not versions:
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_id}")
    return versions
