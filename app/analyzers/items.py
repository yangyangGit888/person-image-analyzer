"""把模型局部结果构建为统一 ``ClothingItem``。"""

from __future__ import annotations

from typing import Any

from app.analyzers.models import SideItemPayload
from app.analyzers.parsing import (
    attributes,
    confidence_or,
    evidence_texts,
    normalized_token,
    occlusion_reasons,
    presence_or,
    resolve_person_side,
    tags,
    visibility_or,
)
from app.schemas.analysis_v2 import (
    ClothingItem,
    ConfidenceLevel,
    Evidence,
    EvidenceSource,
    Geometry,
    ItemCategory,
    Presence,
    Side,
    Visibility,
)


def category_from_value(value: Any) -> ItemCategory:
    token = normalized_token(value)
    mapping = {
        "pants": ItemCategory.PANTS,
        "skirt": ItemCategory.SKIRT,
        "one_piece": ItemCategory.ONE_PIECE,
        "hosiery": ItemCategory.HOSIERY,
        "socks": ItemCategory.SOCKS,
        "shoes": ItemCategory.SHOES,
    }
    return mapping.get(token, ItemCategory.UNKNOWN)


def _safe_presence(presence: Presence, visibility: Visibility) -> Presence:
    """不可见区域不能被判为明确未检测到。"""

    if visibility is Visibility.NOT_VISIBLE and presence is not Presence.DETECTED:
        return Presence.NOT_VISIBLE
    if visibility is Visibility.OCCLUDED and presence is Presence.NOT_DETECTED:
        return Presence.UNCERTAIN
    return presence


def build_evidence(
    payload: SideItemPayload,
    *,
    geometry: Geometry | None,
    source: EvidenceSource,
    fallback: str,
) -> list[Evidence]:
    observations = evidence_texts(payload.evidence)
    if not observations:
        observations = [fallback]
    return [
        Evidence(
            frame_id=geometry.frame_id if geometry else None,
            timestamp=geometry.timestamp if geometry else None,
            bbox=geometry,
            observation=observation,
            confidence=confidence_or(payload.confidence, ConfidenceLevel.LOW),
            source=source,
        )
        for observation in observations[:3]
    ]


def build_item(
    payload: SideItemPayload,
    *,
    category: ItemCategory,
    side: Side,
    analysis_id: str | None,
    source_id: str | None,
    geometry: Geometry | None,
    source: EvidenceSource = EvidenceSource.CROP,
    fallback_observation: str = "模型观察",
    image_side: Side | None = None,
) -> ClothingItem:
    resolved_side, _, _ = resolve_person_side(payload, side)
    visibility = visibility_or(payload.visibility, Visibility.NOT_VISIBLE)
    presence = _safe_presence(
        presence_or(payload.presence, Presence.UNCERTAIN),
        visibility,
    )
    item = ClothingItem(
        analysis_id=analysis_id,
        source_id=source_id,
        category=category,
        type=normalized_token(payload.type),
        side=resolved_side,
        image_side=image_side or side,
        presence=presence,
        visibility=visibility,
        attributes=attributes(payload.attributes),
        confidence=confidence_or(payload.confidence, ConfidenceLevel.LOW),
        geometry=geometry,
        evidence=build_evidence(
            payload,
            geometry=geometry,
            source=source,
            fallback=fallback_observation,
        ),
        occlusion_reason=occlusion_reasons(payload.occlusion_reason),
    )
    item.ensure_type_allowed()
    return item


def build_lower_body_item(
    payload: SideItemPayload,
    *,
    analysis_id: str | None,
    source_id: str | None,
    geometry: Geometry | None,
    source: EvidenceSource,
    style_tags: list[Any] | None = None,
    design_features: list[Any] | None = None,
    fallback_observation: str = "模型观察下装",
) -> ClothingItem:
    item = build_item(
        payload,
        category=category_from_value(getattr(payload, "category", "unknown")),
        side=Side.NOT_APPLICABLE,
        analysis_id=analysis_id,
        source_id=source_id,
        geometry=geometry,
        source=source,
        fallback_observation=fallback_observation,
    )
    item.style_tags = tags(style_tags or [])
    item.design_features = tags(design_features or [])
    return item


def not_visible_item(
    *,
    category: ItemCategory,
    item_type: str,
    side: Side,
    analysis_id: str | None,
    source_id: str | None,
    reason: str,
    image_side: Side | None = None,
) -> ClothingItem:
    item = ClothingItem(
        analysis_id=analysis_id,
        source_id=source_id,
        category=category,
        type=item_type,
        side=side,
        image_side=image_side or side,
        presence=Presence.NOT_VISIBLE,
        visibility=Visibility.NOT_VISIBLE,
        confidence=ConfidenceLevel.LOW,
    )
    item.occlusion_reason = occlusion_reasons([reason])
    return item
