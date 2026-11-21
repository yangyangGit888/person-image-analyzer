"""把全图概览和物品结果合并为身体可观察性。

核心规则：物品是否存在与区域是否可观察分开处理。长裙/长裤遮住腿脚时，
腿脚保持 ``not_visible``，不能推导出“没穿袜子/鞋子”。
"""

from __future__ import annotations

from app.analyzers.models import (
    OverviewBodyPayload,
    OverviewLowerBodyPayload,
    OverviewPayload,
    OverviewRegionPayload,
)
from app.analyzers.parsing import (
    confidence_or,
    normalized_token,
    occlusion_reasons,
    visibility_or,
)
from app.schemas.analysis_v2 import (
    BodyObservability,
    BodyRegion,
    ClothingItem,
    ConfidenceLevel,
    Evidence,
    EvidenceSource,
    OcclusionReason,
    RegionObservability,
    Visibility,
)


def _region(
    region: BodyRegion,
    payload: OverviewRegionPayload,
    *,
    fallback_reasons: list[OcclusionReason] | None = None,
) -> RegionObservability:
    reasons = occlusion_reasons(payload.occlusion_reason)
    if fallback_reasons:
        for reason in fallback_reasons:
            if reason not in reasons:
                reasons.append(reason)
    evidence = [
        Evidence(
            observation=f"全图概览：{region.value} 区域可观察性判断",
            confidence=confidence_or(payload.confidence, ConfidenceLevel.LOW),
            source=EvidenceSource.OVERVIEW,
        )
    ]
    return RegionObservability(
        region=region,
        visibility=visibility_or(payload.visibility, Visibility.NOT_VISIBLE),
        occlusion_reason=reasons,
        confidence=confidence_or(payload.confidence, ConfidenceLevel.LOW),
        evidence=evidence,
    )


def _is_long_lower_body(
    lower_body: OverviewLowerBodyPayload,
    items: list[ClothingItem],
) -> tuple[bool, OcclusionReason | None]:
    for item in items:
        if item.presence.value != "detected":
            continue
        length = normalized_token(item.attributes.length)
        item_type = normalized_token(item.type)
        if item.category.value in {"pants"}:
            if length in {"long", "full", "maxi"} or item_type not in {
                "shorts",
                "hot_pants",
                "unknown",
            }:
                return True, OcclusionReason.LONG_PANTS
        if item.category.value in {"skirt", "one_piece"}:
            if length in {"long", "maxi", "full"} or item_type in {
                "maxi_skirt",
                "midi_skirt",
                "maxi_dress",
                "midi_dress",
            }:
                return True, OcclusionReason.LONG_SKIRT
    return False, None


def _apply_long_garment_rule(
    observability: BodyObservability,
    reason: OcclusionReason,
) -> None:
    for region in (
        BodyRegion.LEFT_LEG,
        BodyRegion.RIGHT_LEG,
        BodyRegion.LEFT_FOOT,
        BodyRegion.RIGHT_FOOT,
    ):
        current = getattr(observability, region.value)
        if current.visibility in (Visibility.CLEAR, Visibility.PARTIAL):
            continue
        current.visibility = Visibility.NOT_VISIBLE
        if reason not in current.occlusion_reason:
            current.occlusion_reason.append(reason)


def build_body_observability(
    overview: OverviewBodyPayload,
    lower_body: OverviewLowerBodyPayload,
    items: list[ClothingItem],
) -> BodyObservability:
    result = BodyObservability(
        lower_body=_region(BodyRegion.LOWER_BODY, overview.lower_body),
        left_leg=_region(BodyRegion.LEFT_LEG, overview.left_leg),
        right_leg=_region(BodyRegion.RIGHT_LEG, overview.right_leg),
        left_foot=_region(BodyRegion.LEFT_FOOT, overview.left_foot),
        right_foot=_region(BodyRegion.RIGHT_FOOT, overview.right_foot),
    )
    is_long, reason = _is_long_lower_body(lower_body, items)
    if is_long and reason is not None:
        _apply_long_garment_rule(result, reason)
    return result


def build_from_payload(
    payload: OverviewPayload,
    items: list[ClothingItem],
) -> BodyObservability:
    return build_body_observability(
        payload.body_observability,
        payload.lower_body,
        items,
    )
