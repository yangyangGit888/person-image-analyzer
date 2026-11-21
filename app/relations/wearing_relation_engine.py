"""穿戴关系规则引擎。

优先使用 Python 规则校验模型结果，不把丝袜、普通袜子、鞋子的叠加关系
完全交给 AI 自由推断。不可见或不确定时返回 unknown，不强行生成关系。
"""

from __future__ import annotations

from app.schemas.analysis_v2 import (
    ClothingItem,
    ConfidenceLevel,
    Evidence,
    EvidenceSource,
    ItemCategory,
    Presence,
    RelationStatus,
    RelationType,
    Side,
    Visibility,
    WearingRelation,
)


def _items_for(
    items: list[ClothingItem],
    category: ItemCategory,
    side: Side,
) -> list[ClothingItem]:
    return [
        item
        for item in items
        if item.category is category and item.side in (side, Side.BOTH)
    ]


def _item_status(item: ClothingItem | None) -> RelationStatus:
    if item is None:
        return RelationStatus.UNKNOWN
    if item.visibility in (Visibility.NOT_VISIBLE,) or item.presence in (
        Presence.NOT_VISIBLE,
    ):
        return RelationStatus.UNKNOWN
    if item.presence in (Presence.UNCERTAIN,):
        return RelationStatus.UNKNOWN
    if item.presence is Presence.DETECTED:
        return RelationStatus.PRESENT
    if item.presence is Presence.NOT_DETECTED:
        return RelationStatus.ABSENT
    return RelationStatus.UNKNOWN


def _pair_status(
    first: ClothingItem | None,
    second: ClothingItem | None,
) -> RelationStatus:
    first_status = _item_status(first)
    second_status = _item_status(second)
    if RelationStatus.UNKNOWN in (first_status, second_status):
        return RelationStatus.UNKNOWN
    if first_status is RelationStatus.PRESENT and second_status is RelationStatus.PRESENT:
        return RelationStatus.PRESENT
    return RelationStatus.ABSENT


def _relation(
    relation_type: RelationType,
    side: Side,
    status: RelationStatus,
    *,
    observation: str,
) -> WearingRelation:
    return WearingRelation(
        type=relation_type,
        side=side,
        status=status,
        evidence=[
            Evidence(
                observation=observation,
                confidence=ConfidenceLevel.MEDIUM
                if status is RelationStatus.PRESENT
                else ConfidenceLevel.LOW,
                source=EvidenceSource.RULE,
            )
        ],
    )


def _hosiery_covers_feet_status(item: ClothingItem | None) -> RelationStatus:
    status = _item_status(item)
    if status is not RelationStatus.PRESENT:
        return status
    assert item is not None
    if item.attributes.covers_feet is True or item.type == "pantyhose":
        return RelationStatus.PRESENT
    if item.attributes.covers_feet is False:
        return RelationStatus.ABSENT
    return RelationStatus.UNKNOWN


def build_wearing_relations(items: list[ClothingItem]) -> list[WearingRelation]:
    relations: list[WearingRelation] = []
    for side in (Side.LEFT, Side.RIGHT):
        hosiery = _items_for(items, ItemCategory.HOSIERY, side)
        socks = _items_for(items, ItemCategory.SOCKS, side)
        shoes = _items_for(items, ItemCategory.SHOES, side)

        hosiery_item = hosiery[0] if hosiery else None
        socks_item = socks[0] if socks else None
        shoes_item = shoes[0] if shoes else None

        relations.append(
            _relation(
                RelationType.HOSIERY_COVERS_FEET,
                side,
                _hosiery_covers_feet_status(hosiery_item),
                observation=f"{side.value} 侧丝袜是否覆盖脚部",
            )
        )
        relations.append(
            _relation(
                RelationType.REGULAR_SOCKS_OVER_HOSIERY,
                side,
                _pair_status(socks_item, hosiery_item),
                observation=f"{side.value} 侧普通袜子与丝袜叠加关系",
            )
        )
        relations.append(
            _relation(
                RelationType.SHOES_OVER_SOCKS,
                side,
                _pair_status(shoes_item, socks_item),
                observation=f"{side.value} 侧鞋子与普通袜子叠加关系",
            )
        )
        relations.append(
            _relation(
                RelationType.SHOES_OVER_HOSIERY,
                side,
                _pair_status(shoes_item, hosiery_item),
                observation=f"{side.value} 侧鞋子与丝袜叠加关系",
            )
        )
    return relations
