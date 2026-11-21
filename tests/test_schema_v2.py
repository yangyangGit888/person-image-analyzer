"""V2 数据结构与旧响应兼容适配测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.analysis_v2 import (
    AnalysisResultV2,
    BodyObservability,
    BodyRegion,
    ClothingItem,
    ConfidenceLevel,
    FootState,
    Geometry,
    ItemCategory,
    OcclusionReason,
    Presence,
    RegionObservability,
    Side,
    Visibility,
)
from app.schemas.analysis_v2 import (
    FootState as FootStateModel,
)
from app.services.legacy_adapter import to_legacy_persons


def _item(
    category: ItemCategory,
    item_type: str,
    side: Side,
    presence: Presence = Presence.DETECTED,
    visibility: Visibility = Visibility.CLEAR,
    color: str | None = None,
) -> ClothingItem:
    return ClothingItem(
        category=category,
        type=item_type,
        side=side,
        presence=presence,
        visibility=visibility,
        attributes={"color": color} if color else {},
        confidence=ConfidenceLevel.HIGH,
    )


def test_geometry_requires_valid_bbox() -> None:
    geometry = Geometry(source_width=100, source_height=200, bbox=[10, 20, 50, 80])
    assert geometry.bbox == [10, 20, 50, 80]

    with pytest.raises(ValidationError):
        Geometry(source_width=100, source_height=100, bbox=[90, 90, 120, 120])
    with pytest.raises(ValidationError):
        Geometry(source_width=100, source_height=100, bbox=[50, 50, 40, 80])


def test_item_type_is_normalized_and_validated() -> None:
    item = ClothingItem(category=ItemCategory.PANTS, type=" Pencil_Skirt ")
    item.ensure_type_allowed()
    assert item.type == "unknown"

    pants = ClothingItem(category=ItemCategory.PANTS, type="Yoga_Pants")
    pants.ensure_type_allowed()
    assert pants.type == "yoga_pants"

    tagged = ClothingItem(
        category=ItemCategory.SKIRT,
        type="pleated_skirt",
        style_tags=["JK Style", "jk_style"],
    )
    assert tagged.style_tags == ["jk_style"]


def test_long_skirt_does_not_become_not_detected() -> None:
    result = AnalysisResultV2(
        person_count=1,
        items=[
            _item(ItemCategory.SKIRT, "maxi_skirt", Side.NOT_APPLICABLE),
        ],
        body_observability=BodyObservability(
            lower_body=RegionObservability(
                region=BodyRegion.LOWER_BODY,
                visibility=Visibility.PARTIAL,
                occlusion_reason=[OcclusionReason.LONG_SKIRT],
            ),
            left_leg=RegionObservability(
                region=BodyRegion.LEFT_LEG,
                visibility=Visibility.NOT_VISIBLE,
                occlusion_reason=[OcclusionReason.LONG_SKIRT],
            ),
            right_leg=RegionObservability(
                region=BodyRegion.RIGHT_LEG,
                visibility=Visibility.NOT_VISIBLE,
                occlusion_reason=[OcclusionReason.LONG_SKIRT],
            ),
            left_foot=RegionObservability(
                region=BodyRegion.LEFT_FOOT,
                visibility=Visibility.NOT_VISIBLE,
                occlusion_reason=[OcclusionReason.LONG_SKIRT],
            ),
            right_foot=RegionObservability(
                region=BodyRegion.RIGHT_FOOT,
                visibility=Visibility.NOT_VISIBLE,
                occlusion_reason=[OcclusionReason.LONG_SKIRT],
            ),
        ),
        feet=[
            FootState(side=Side.LEFT, barefoot="not_visible", visibility=Visibility.NOT_VISIBLE),
            FootState(side=Side.RIGHT, barefoot="not_visible", visibility=Visibility.NOT_VISIBLE),
        ],
    )

    person = to_legacy_persons(result)[0]
    assert person.lower_body.wearing_skirt == "true"
    assert person.feet.wearing_socks == "unknown"
    assert person.feet.wearing_shoes == "unknown"
    assert person.feet.barefoot == "unknown"
    assert person.visibility.feet_visible == "false"


def test_socks_can_exist_over_hosiery_and_under_shoes() -> None:
    result = AnalysisResultV2(
        person_count=1,
        items=[
            _item(ItemCategory.HOSIERY, "pantyhose", Side.LEFT, color="black"),
            _item(ItemCategory.SOCKS, "ankle", Side.LEFT, color="white"),
            _item(ItemCategory.SHOES, "sneakers", Side.LEFT, color="white"),
        ],
        feet=[
            FootStateModel(
                side=Side.LEFT,
                barefoot="not_barefoot",
                visibility=Visibility.CLEAR,
            )
        ],
    )

    person = to_legacy_persons(result)[0]
    assert person.legs.wearing_tights == "true"
    assert person.feet.wearing_socks == "true"
    assert person.feet.socks_type == "短袜"
    assert person.feet.wearing_shoes == "true"
    assert person.feet.barefoot == "false"


def test_left_and_right_items_keep_their_side() -> None:
    left_shoe = _item(
        ItemCategory.SHOES,
        "boots",
        Side.LEFT,
        presence=Presence.DETECTED,
    )
    right_shoe = _item(
        ItemCategory.SHOES,
        "unknown",
        Side.RIGHT,
        presence=Presence.NOT_DETECTED,
        visibility=Visibility.CLEAR,
    )

    assert left_shoe.side is Side.LEFT
    assert right_shoe.side is Side.RIGHT
    assert left_shoe.presence is Presence.DETECTED
    assert right_shoe.presence is Presence.NOT_DETECTED


def test_barefoot_requires_visible_bare_feet() -> None:
    result = AnalysisResultV2(
        person_count=1,
        feet=[
            FootStateModel(
                side=Side.LEFT,
                barefoot="barefoot",
                visibility=Visibility.CLEAR,
            ),
            FootStateModel(
                side=Side.RIGHT,
                barefoot="barefoot",
                visibility=Visibility.CLEAR,
            ),
        ],
    )

    person = to_legacy_persons(result)[0]
    assert person.feet.barefoot == "true"
