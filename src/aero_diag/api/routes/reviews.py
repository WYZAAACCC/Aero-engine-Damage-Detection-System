"""复核 API 路由——发起/提交专家复核。"""

from fastapi import APIRouter, Depends, HTTPException

from aero_diag.api.dependencies import get_review_service
from aero_diag.services.review_service import ReviewService

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.post("/request", status_code=201)
def request_review(
    review_data: dict,
    service: ReviewService = Depends(get_review_service),
) -> dict:
    """创建复核请求。"""
    from aero_diag.domain.decision import DecisionDraft

    try:
        draft = DecisionDraft.model_validate(review_data.get("draft", {}))
        review = service.request_review(
            draft,
            evidence_package_id=review_data.get("evidence_package_id", ""),
        )
        return review.model_dump(mode="json")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{draft_id}/approve")
def approve_review(
    draft_id: str,
    approval_data: dict,
    service: ReviewService = Depends(get_review_service),
) -> dict:
    """批准决策草案。"""
    review = service.approve(
        draft_id=draft_id,
        reviewer=approval_data.get("reviewer", ""),
        reviewer_role=approval_data.get("reviewer_role", ""),
        comments=approval_data.get("comments", ""),
        conditions=approval_data.get("conditions"),
    )
    if review is None:
        raise HTTPException(status_code=404, detail=f"No pending review for draft: {draft_id}")
    return review.model_dump(mode="json")


@router.post("/{draft_id}/reject")
def reject_review(
    draft_id: str,
    rejection_data: dict,
    service: ReviewService = Depends(get_review_service),
) -> dict:
    """驳回决策草案。"""
    review = service.reject(
        draft_id=draft_id,
        reviewer=rejection_data.get("reviewer", ""),
        reviewer_role=rejection_data.get("reviewer_role", ""),
        comments=rejection_data.get("comments", ""),
    )
    if review is None:
        raise HTTPException(status_code=404, detail=f"No pending review for draft: {draft_id}")
    return review.model_dump(mode="json")


@router.get("/{draft_id}/history")
def get_review_history(
    draft_id: str,
    service: ReviewService = Depends(get_review_service),
) -> list[dict]:
    """获取复核历史。"""
    return [r.model_dump(mode="json") for r in service.get_review_history(draft_id)]
