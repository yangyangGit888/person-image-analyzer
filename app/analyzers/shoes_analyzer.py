"""鞋子物品构建。"""

from __future__ import annotations

from app.analyzers.items import build_item, not_visible_item
from app.analyzers.models import FeetSidePayload
from app.analyzers.parsing import resolve_person_side, visibility_or
from app.schemas.analysis_v2 import (
    ClothingItem,
    Geometry,
    ItemCategory,
    Presence,
    Side,
    Visibility,
)


def build_shoes_item(
    side_payload: FeetSidePayload,
    *,
    side: Side,
    analysis_id: str | None,
    source_id: str | None,
    geometry: Geometry | None,
    occlusion_reason: str = "unknown",
) -> ClothingItem:
    resolved_side, _, _ = resolve_person_side(side_payload, side)
    visibility = visibility_or(side_payload.visibility, Visibility.NOT_VISIBLE)
    if visibility is Visibility.NOT_VISIBLE:
        return not_visible_item(
            category=ItemCategory.SHOES,
            item_type="unknown",
            side=resolved_side,
            analysis_id=analysis_id,
            source_id=source_id,
            reason=occlusion_reason,
            image_side=side,
        )
    if side_payload.shoes is None:
        return ClothingItem(
            analysis_id=analysis_id,
            source_id=source_id,
            category=ItemCategory.SHOES,
            type="unknown",
            side=resolved_side,
            image_side=side,
            presence=Presence.UNCERTAIN,
            visibility=visibility,
            geometry=geometry,
        )
    return build_item(
        side_payload.shoes,
        category=ItemCategory.SHOES,
        side=resolved_side,
        analysis_id=analysis_id,
        source_id=source_id,
        geometry=geometry,
        fallback_observation="脚部裁剪图中的鞋子观察",
        image_side=side,
    )
