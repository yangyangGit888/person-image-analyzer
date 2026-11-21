"""V2 分析结果到旧 ``persons`` 响应的兼容适配器。

V2 是主数据结构；旧接口仍返回 ``persons``，这里只做字段映射，不改变 V2
结果，也不把 ``not_visible`` 错误地转换成 ``not_detected``。
"""

from __future__ import annotations

from collections.abc import Iterable

from app.schemas.analysis import (
    AnalysisData,
    FeetAttributes,
    HeadAttributes,
    ImageInfo,
    LegAttributes,
    LowerBodyAttributes,
    PersonAnalysis,
    PersonConfidence,
    Position,
    UpperBodyAttributes,
    Visibility,
)
from app.schemas.analysis_v2 import (
    AnalysisResultV2,
    BarefootStatus,
    BodyRegion,
    ClothingItem,
    HosieryType,
    ItemCategory,
    Presence,
)
from app.schemas.analysis_v2 import (
    Visibility as VisibilityV2,
)

_HOSIERY_TYPES = {
    HosieryType.PANTYHOSE.value: "连裤袜",
    HosieryType.THIGH_HIGH.value: "大腿袜",
    HosieryType.KNEE_HIGH.value: "及膝袜",
    HosieryType.CALF_HIGH.value: "小腿袜",
    HosieryType.ANKLE_HIGH.value: "踝部袜",
    HosieryType.OTHER.value: "其他",
    HosieryType.UNKNOWN.value: "unknown",
}

_SOCKS_TYPES = {
    "no_show": "船袜",
    "ankle": "短袜",
    "crew": "中筒袜",
    "knee_high": "及膝袜",
    "thigh_high": "大腿袜",
    "other": "其他",
    "unknown": "unknown",
}

_SHOES_TYPES = {
    "sneakers": "运动鞋",
    "heels": "高跟鞋",
    "flats": "平底鞋",
    "loafers": "乐福鞋",
    "boots": "靴子",
    "sandals": "凉鞋",
    "slippers": "拖鞋",
    "other": "其他",
    "unknown": "unknown",
}


def _tri_from_presence(presence: Presence) -> str:
    if presence is Presence.DETECTED:
        return "true"
    if presence is Presence.NOT_DETECTED:
        return "false"
    return "unknown"


def _tri_from_visibility(value: VisibilityV2) -> str:
    if value in (VisibilityV2.CLEAR, VisibilityV2.PARTIAL):
        return "true"
    if value is VisibilityV2.NOT_VISIBLE:
        return "false"
    return "unknown"


def _items_of(items: Iterable[ClothingItem], *categories: ItemCategory) -> list[ClothingItem]:
    wanted = set(categories)
    return [item for item in items if item.category in wanted]


def _first(items: Iterable[ClothingItem], category: ItemCategory) -> ClothingItem | None:
    return next((item for item in items if item.category is category), None)


def _any_presence(items: Iterable[ClothingItem], presence: Presence) -> bool:
    return any(item.presence is presence for item in items)


def _all_presence(items: list[ClothingItem], presence: Presence) -> bool:
    return bool(items) and all(item.presence is presence for item in items)


def _item_color(item: ClothingItem | None) -> str:
    if item is None or item.attributes.color in (None, ""):
        return "unknown"
    return str(item.attributes.color)


def _socks_type_text(item: ClothingItem | None) -> str:
    if item is None:
        return "unknown"
    return _SOCKS_TYPES.get(item.type, "unknown")


def _shoes_type_text(item: ClothingItem | None) -> str:
    if item is None:
        return "unknown"
    return _SHOES_TYPES.get(item.type, "unknown")


def _hosiery_type_text(item: ClothingItem | None) -> str:
    if item is None:
        return "unknown"
    return _HOSIERY_TYPES.get(item.type, "unknown")


def _region(result: AnalysisResultV2, region: BodyRegion):
    return getattr(result.body_observability, region.value)


def _barefoot_tri(result: AnalysisResultV2) -> str:
    states = result.feet
    if not states:
        return "unknown"
    if any(state.barefoot is BarefootStatus.BAREFOOT for state in states):
        return "true"
    if states and all(
        state.barefoot is BarefootStatus.NOT_BAREFOOT for state in states
    ):
        return "false"
    return "unknown"


def _legs_visible_tri(result: AnalysisResultV2) -> str:
    values = [
        _region(result, BodyRegion.LEFT_LEG).visibility,
        _region(result, BodyRegion.RIGHT_LEG).visibility,
    ]
    if any(value in (VisibilityV2.CLEAR, VisibilityV2.PARTIAL) for value in values):
        return "true"
    if all(value is VisibilityV2.NOT_VISIBLE for value in values):
        return "false"
    return "unknown"


def _feet_visible_tri(result: AnalysisResultV2) -> str:
    values = [
        _region(result, BodyRegion.LEFT_FOOT).visibility,
        _region(result, BodyRegion.RIGHT_FOOT).visibility,
    ]
    if any(value in (VisibilityV2.CLEAR, VisibilityV2.PARTIAL) for value in values):
        return "true"
    if all(value is VisibilityV2.NOT_VISIBLE for value in values):
        return "false"
    return "unknown"


def _leg_covering(items: list[ClothingItem]) -> str:
    parts: list[str] = []
    for item in items:
        if item.presence is not Presence.DETECTED:
            continue
        label = item.type
        color = item.attributes.color
        if color:
            label = f"{label}({color})"
        parts.append(label)
    return "、".join(parts) if parts else "unknown"


def _build_lower_body(items: list[ClothingItem]) -> LowerBodyAttributes:
    pants = _first(items, ItemCategory.PANTS)
    skirt = _first(items, ItemCategory.SKIRT)
    one_piece = _first(items, ItemCategory.ONE_PIECE)

    if one_piece is not None:
        pants_tri = "false" if one_piece.presence is Presence.DETECTED else "unknown"
        skirt_tri = _tri_from_presence(one_piece.presence)
        skirt_color = _item_color(one_piece)
    else:
        pants_tri = _tri_from_presence(pants.presence) if pants else "unknown"
        skirt_tri = _tri_from_presence(skirt.presence) if skirt else "unknown"
        skirt_color = _item_color(skirt)

    pants_type = "unknown"
    if pants is not None and pants.presence is Presence.DETECTED:
        pants_type = pants.type

    return LowerBodyAttributes(
        wearing_pants=pants_tri,  # type: ignore[arg-type]
        pants_type=pants_type,
        pants_color=_item_color(pants),
        wearing_skirt=skirt_tri,  # type: ignore[arg-type]
        skirt_color=skirt_color,
    )


def _build_legs(items: list[ClothingItem], result: AnalysisResultV2) -> LegAttributes:
    hosiery = _items_of(items, ItemCategory.HOSIERY)
    detected_hosiery = [item for item in hosiery if item.presence is Presence.DETECTED]
    not_detected_hosiery = _all_presence(hosiery, Presence.NOT_DETECTED)

    if any(item.type == HosieryType.PANTYHOSE.value for item in detected_hosiery):
        tights = "true"
    elif not_detected_hosiery:
        tights = "false"
    else:
        tights = "unknown"

    other_hosiery = [
        item
        for item in detected_hosiery
        if item.type != HosieryType.PANTYHOSE.value
    ]
    if other_hosiery:
        stockings = "true"
    elif not_detected_hosiery:
        stockings = "false"
    else:
        stockings = "unknown"

    pants = _first(items, ItemCategory.PANTS)
    shorts = (
        pants is not None
        and pants.presence is Presence.DETECTED
        and pants.type == "shorts"
    )
    barefoot = _barefoot_tri(result)
    skin_visible = "true" if barefoot == "true" else "unknown"

    return LegAttributes(
        skin_visible=skin_visible,  # type: ignore[arg-type]
        wearing_tights=tights,  # type: ignore[arg-type]
        wearing_stockings=stockings,  # type: ignore[arg-type]
        wearing_long_pants="true" if pants is not None and pants.presence is Presence.DETECTED else "unknown",
        wearing_shorts="true" if shorts else "unknown",
        leg_covering=_leg_covering(hosiery + _items_of(items, ItemCategory.PANTS, ItemCategory.SKIRT, ItemCategory.ONE_PIECE)),
    )


def _build_feet(items: list[ClothingItem], result: AnalysisResultV2) -> FeetAttributes:
    socks = _items_of(items, ItemCategory.SOCKS)
    shoes = _items_of(items, ItemCategory.SHOES)
    detected_socks = [item for item in socks if item.presence is Presence.DETECTED]
    detected_shoes = [item for item in shoes if item.presence is Presence.DETECTED]

    if detected_socks:
        socks_tri = "true"
    elif _all_presence(socks, Presence.NOT_DETECTED):
        socks_tri = "false"
    else:
        socks_tri = "unknown"

    if detected_shoes:
        shoes_tri = "true"
    elif _all_presence(shoes, Presence.NOT_DETECTED):
        shoes_tri = "false"
    else:
        shoes_tri = "unknown"

    socks_item = detected_socks[0] if detected_socks else None
    shoes_item = detected_shoes[0] if detected_shoes else None

    return FeetAttributes(
        wearing_socks=socks_tri,  # type: ignore[arg-type]
        socks_type=_socks_type_text(socks_item),
        socks_color=_item_color(socks_item),
        wearing_shoes=shoes_tri,  # type: ignore[arg-type]
        shoes_type=_shoes_type_text(shoes_item),
        shoes_color=_item_color(shoes_item),
        barefoot=_barefoot_tri(result),  # type: ignore[arg-type]
    )


def _build_evidence(result: AnalysisResultV2) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for warning in result.warnings[:10]:
        evidence[warning.code[:50]] = warning.message[:200]
    for item in result.items[:10]:
        if not item.evidence:
            continue
        observation = item.evidence[0].observation.strip()
        if observation:
            evidence[item.item_id[:50]] = observation[:200]
    return evidence


def to_legacy_persons(result: AnalysisResultV2) -> list[PersonAnalysis]:
    """把 V2 结果转换为旧 ``persons`` 结构。"""

    if result.person_count <= 0 and not result.items:
        return []

    items = result.normalized_items()
    lower_body = _build_lower_body(items)
    legs = _build_legs(items, result)
    feet = _build_feet(items, result)

    return [
        PersonAnalysis(
            person_id=1,
            position=Position(description="图片中的人物", bbox=None),
            visibility=Visibility(
                full_body_visible="unknown",
                head_visible="unknown",
                upper_body_visible="unknown",
                legs_visible=_legs_visible_tri(result),  # type: ignore[arg-type]
                feet_visible=_feet_visible_tri(result),  # type: ignore[arg-type]
            ),
            head=HeadAttributes(),
            upper_body=UpperBodyAttributes(),
            lower_body=lower_body,
            legs=legs,
            feet=feet,
            confidence=PersonConfidence(),
            evidence=_build_evidence(result),
        )
    ]


def to_legacy_analysis_data(
    result: AnalysisResultV2,
    *,
    width: int,
    height: int,
) -> AnalysisData:
    persons = to_legacy_persons(result)
    return AnalysisData(
        image_info=ImageInfo(
            width=width,
            height=height,
            person_count=len(persons),
        ),
        persons=persons,
        schema_version=2,
        analysis_id=result.analysis_id,
        person_count=result.person_count,
        source_type=result.source_type,
        status=result.status,
        person_orientation=result.person_orientation,
        side_basis=result.side_basis,
        side_assignment_confidence=result.side_assignment_confidence,
        items=result.items,
        body_observability=result.body_observability,
        feet=result.feet,
        wearing_relations=result.wearing_relations,
        evidence=result.evidence,
        warnings=result.warnings,
        created_at=result.created_at,
        prompt_versions=result.prompt_versions,
        model=result.model,
        usage=result.usage,
    )
