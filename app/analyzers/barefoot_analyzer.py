"""赤脚状态规则。

模型只提供观察；最终状态必须结合脚部可见性、丝袜/普通袜子/鞋子物品和
是否有覆盖证据。丝袜、袜子或鞋子存在时，绝不输出 barefoot。
"""

from __future__ import annotations

from app.analyzers.models import FeetSidePayload
from app.analyzers.parsing import (
    barefoot_or,
    confidence_or,
    evidence_texts,
    optional_bool,
    resolve_person_side,
    visibility_or,
)
from app.schemas.analysis_v2 import (
    BarefootStatus,
    ClothingItem,
    ConfidenceLevel,
    Evidence,
    EvidenceSource,
    FootState,
    Presence,
    Side,
    Visibility,
)


def _cover_items(
    items: list[ClothingItem],
    side: Side,
) -> list[ClothingItem]:
    covering: list[ClothingItem] = []
    for item in items:
        if item.side not in (side, Side.BOTH):
            continue
        if item.presence is not Presence.DETECTED:
            continue
        if item.category.value not in {"hosiery", "socks", "shoes"}:
            continue
        if (
            item.category.value == "hosiery"
            and item.attributes.covers_feet is False
        ):
            continue
        covering.append(item)
    return covering


def resolve_foot_state(
    *,
    side: Side,
    side_payload: FeetSidePayload,
    body_visibility: Visibility,
    items: list[ClothingItem],
) -> tuple[FootState, list[str]]:
    warnings: list[str] = []
    resolved_side, _, _ = resolve_person_side(side_payload, side)
    visibility = visibility_or(side_payload.visibility, body_visibility)
    if body_visibility is Visibility.NOT_VISIBLE:
        visibility = Visibility.NOT_VISIBLE

    model_barefoot = barefoot_or(side_payload.barefoot, BarefootStatus.UNKNOWN)
    cover_items = _cover_items(items, resolved_side)
    skin_visible = optional_bool(side_payload.skin_visible)

    if visibility is Visibility.NOT_VISIBLE:
        status = BarefootStatus.NOT_VISIBLE
    elif cover_items:
        status = BarefootStatus.NOT_BAREFOOT
        if model_barefoot is BarefootStatus.BAREFOOT:
            warnings.append(
                f"{side.value} 脚存在丝袜/袜子/鞋子覆盖，模型赤脚判断已覆盖"
            )
    elif model_barefoot is BarefootStatus.BAREFOOT:
        status = (
            BarefootStatus.BAREFOOT
            if skin_visible is not False
            else BarefootStatus.UNKNOWN
        )
    elif model_barefoot is BarefootStatus.NOT_BAREFOOT:
        status = BarefootStatus.NOT_BAREFOOT
    else:
        status = BarefootStatus.UNKNOWN

    evidence = [
        Evidence(
            observation=text,
            confidence=confidence_or(
                side_payload.confidence,
                ConfidenceLevel.LOW,
            ),
            source=EvidenceSource.CROP,
        )
        for text in evidence_texts(side_payload.evidence)[:3]
    ]
    return (
        FootState(
            side=resolved_side,
            image_side=side,
            barefoot=status,
            visibility=visibility,
            confidence=confidence_or(
                side_payload.confidence,
                ConfidenceLevel.LOW,
            ),
            item_ids=(
                [item.item_id for item in cover_items]
                if resolved_side is not Side.UNKNOWN
                else []
            ),
            evidence=evidence,
            uncertainty_reason=(
                "脚部不可见"
                if status is BarefootStatus.NOT_VISIBLE
                else "赤脚证据不足"
                if status is BarefootStatus.UNKNOWN
                else None
            ),
        ),
        warnings,
    )
