"""复核服务——专家复核请求、批准、驳回和签署。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from aero_diag.domain.decision import DecisionDraft, ReviewDecision


class ReviewService:
    """专家复核服务——管理所有审批流程。"""

    def __init__(self) -> None:
        self._reviews: dict[str, list[ReviewDecision]] = {}  # draft_id -> [versions]

    def request_review(
        self,
        draft: DecisionDraft,
        *,
        evidence_package_id: str = "",
    ) -> ReviewDecision:
        """创建复核请求。"""
        review = ReviewDecision(
            review_id=uuid.uuid4().hex[:12],
            draft_id=draft.draft_id,
            task_id=draft.task_id,
            decision="",
            comments="",
            evidence_package_id=evidence_package_id,
            version=1,
        )
        if draft.draft_id not in self._reviews:
            self._reviews[draft.draft_id] = []
        self._reviews[draft.draft_id].append(review)
        return review

    def approve(
        self,
        draft_id: str,
        *,
        reviewer: str,
        reviewer_role: str,
        comments: str = "",
        conditions: list[str] | None = None,
    ) -> ReviewDecision | None:
        """批准决策草案。"""
        reviews = self._reviews.get(draft_id, [])
        if not reviews:
            return None

        latest = reviews[-1]
        new_review = ReviewDecision(
            review_id=uuid.uuid4().hex[:12],
            draft_id=draft_id,
            task_id=latest.task_id,
            decision="approved",
            comments=comments,
            conditions=conditions or [],
            reviewer=reviewer,
            reviewer_role=reviewer_role,
            evidence_package_id=latest.evidence_package_id,
            version=latest.version + 1,
            supersedes_review_id=latest.review_id,
        )
        self._reviews[draft_id].append(new_review)
        return new_review

    def reject(
        self,
        draft_id: str,
        *,
        reviewer: str,
        reviewer_role: str,
        comments: str = "",
    ) -> ReviewDecision | None:
        """驳回决策草案。"""
        reviews = self._reviews.get(draft_id, [])
        if not reviews:
            return None

        latest = reviews[-1]
        new_review = ReviewDecision(
            review_id=uuid.uuid4().hex[:12],
            draft_id=draft_id,
            task_id=latest.task_id,
            decision="rejected",
            comments=comments,
            reviewer=reviewer,
            reviewer_role=reviewer_role,
            evidence_package_id=latest.evidence_package_id,
            version=latest.version + 1,
            supersedes_review_id=latest.review_id,
        )
        self._reviews[draft_id].append(new_review)
        return new_review

    def get_review_history(self, draft_id: str) -> list[ReviewDecision]:
        """获取复核历史（所有版本）。"""
        return list(self._reviews.get(draft_id, []))

    def get_latest_review(self, draft_id: str) -> ReviewDecision | None:
        """获取最新复核决定。"""
        reviews = self._reviews.get(draft_id, [])
        return reviews[-1] if reviews else None
