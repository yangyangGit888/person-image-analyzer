"""模型字段安全归一化与局部坐标映射。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, TypeVar

from app.schemas.analysis_v2 import (
    BarefootStatus,
    BodyRegion,
    ConfidenceLevel,
    Geometry,
    GeometryPrecision,
    ItemAttributes,
    OcclusionReason,
    Presence,
    Side,
    SideBasis,
    Visibility,
)
from app.utils.coords import pixel_bbox_validate

EnumT = TypeVar("EnumT", bound=Enum)

_UNKNOWN_TEXTS = {
    "",
    "unknown",
    "none",
    "null",
    "n/a",
    "na",
    "undefined",
    "未知",
    "无法判断",
    "无法识别",
    "不清楚",
    "不确定",
}

_ALIASES: dict[str, str] = {
    "连裤袜": "pantyhose",
    "大腿袜": "thigh_high",
    "及膝袜": "knee_high",
    "小腿袜": "calf_high",
    "踝部袜": "ankle_high",
    "船袜": "no_show",
    "短袜": "ankle",
    "中筒袜": "crew",
    "长筒袜": "thigh_high",
    "运动鞋": "sneakers",
    "高跟鞋": "heels",
    "平底鞋": "flats",
    "乐福鞋": "loafers",
    "靴子": "boots",
    "凉鞋": "sandals",
    "拖鞋": "slippers",
    "裤子": "pants",
    "裙": "skirt",
    "半身裙": "skirt",
    "连衣裙": "one_piece",
    "正面": "front",
    "背面": "back",
    "左侧面": "left_profile",
    "右侧面": "right_profile",
    "四分之三侧面": "three_quarter",
}


def normalize_text(value: Any, *, default: str = "unknown") -> str:
    if not isinstance(value, str):
        return default
    text = value.strip().strip('"').strip("'").strip()
    if not text or text.lower() in _UNKNOWN_TEXTS or text in _UNKNOWN_TEXTS:
        return default
    text = " ".join(text.split())
    return text[:120]


def normalized_token(value: Any) -> str:
    text = normalize_text(value)
    if text == "unknown":
        return "unknown"
    lowered = text.lower().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(text, _ALIASES.get(lowered, lowered))


def enum_or(value: Any, enum_type: type[EnumT], default: EnumT) -> EnumT:
    token = normalized_token(value)
    try:
        return enum_type(token)
    except ValueError:
        return default


def presence_or(value: Any, default: Presence = Presence.UNCERTAIN) -> Presence:
    return enum_or(value, Presence, default)


def visibility_or(
    value: Any,
    default: Visibility = Visibility.NOT_VISIBLE,
) -> Visibility:
    return enum_or(value, Visibility, default)


def confidence_or(
    value: Any,
    default: ConfidenceLevel = ConfidenceLevel.LOW,
) -> ConfidenceLevel:
    return enum_or(value, ConfidenceLevel, default)


def barefoot_or(
    value: Any,
    default: BarefootStatus = BarefootStatus.UNKNOWN,
) -> BarefootStatus:
    return enum_or(value, BarefootStatus, default)


def side_or(value: Any, default: Side = Side.UNKNOWN) -> Side:
    return enum_or(value, Side, default)


def side_basis_or(
    value: Any,
    default: SideBasis = SideBasis.UNKNOWN,
) -> SideBasis:
    return enum_or(value, SideBasis, default)


def resolve_person_side(
    payload: Any,
    fallback: Side,
) -> tuple[Side, ConfidenceLevel, SideBasis]:
    """解析人物自身左右；缺少新字段时兼容旧结果。

    模型明确使用画面左右或低置信度时返回 unknown，不把图像左右当成人物
    自身左右。
    """

    raw_person_side = getattr(payload, "person_side", None)
    raw_basis = getattr(payload, "side_basis", None)
    raw_confidence = getattr(payload, "side_confidence", None)
    if raw_person_side is None and raw_basis is None and raw_confidence is None:
        return fallback, ConfidenceLevel.MEDIUM, SideBasis.PERSON_OWN

    explicit = side_or(raw_person_side, Side.UNKNOWN)
    basis = side_basis_or(raw_basis, SideBasis.UNKNOWN)
    confidence = confidence_or(raw_confidence, ConfidenceLevel.LOW)

    if explicit in (Side.LEFT, Side.RIGHT):
        if basis is SideBasis.IMAGE_SIDE:
            return Side.UNKNOWN, ConfidenceLevel.LOW, basis
        if confidence is ConfidenceLevel.LOW and basis is SideBasis.UNKNOWN:
            return Side.UNKNOWN, ConfidenceLevel.LOW, basis
        return explicit, confidence, (
            basis if basis is not SideBasis.UNKNOWN else SideBasis.PERSON_OWN
        )

    if fallback in (Side.LEFT, Side.RIGHT):
        if basis is SideBasis.IMAGE_SIDE or confidence is ConfidenceLevel.LOW:
            return Side.UNKNOWN, ConfidenceLevel.LOW, basis
        return fallback, confidence, basis
    return Side.UNKNOWN, ConfidenceLevel.LOW, basis


def occlusion_reasons(value: Any) -> list[OcclusionReason]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result: list[OcclusionReason] = []
    for item in value:
        reason = enum_or(item, OcclusionReason, OcclusionReason.UNKNOWN)
        if reason not in result:
            result.append(reason)
    return result


def tags(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip().lower().replace(" ", "_")
        if text and text not in result:
            result.append(text[:60])
    return result


def warnings(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result: list[str] = []
    for item in value:
        text = normalize_text(item)
        if text != "unknown":
            result.append(text[:200])
    return result


def evidence_texts(value: Any) -> list[str]:
    return warnings(value)


def attributes(value: Any) -> ItemAttributes:
    if not isinstance(value, Mapping):
        return ItemAttributes()
    try:
        return ItemAttributes.model_validate(dict(value))
    except (TypeError, ValueError):
        return ItemAttributes()


def optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "是"}:
            return True
        if normalized in {"false", "no", "n", "0", "否"}:
            return False
    return None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def normalized_box(value: Any) -> tuple[float, float, float, float] | None:
    """解析归一化框。

    Mapping 按 ``x/y/width/height`` 或 ``x_min/y_min/x_max/y_max``；
    Sequence 固定按 ``[x_min, y_min, x_max, y_max]``。
    """

    if isinstance(value, Mapping):
        x = _finite(value.get("x", value.get("x_min")))
        y = _finite(value.get("y", value.get("y_min")))
        width = _finite(value.get("width", value.get("w")))
        height = _finite(value.get("height", value.get("h")))
        if all(item is not None for item in (x, y, width, height)):
            assert x is not None and y is not None
            assert width is not None and height is not None
            if width > 0 and height > 0:
                return (x, y, width, height)
        x2 = _finite(value.get("x_max"))
        y2 = _finite(value.get("y_max"))
        if all(item is not None for item in (x, y, x2, y2)):
            assert x is not None and y is not None
            assert x2 is not None and y2 is not None
            if x2 > x and y2 > y:
                return (x, y, x2 - x, y2 - y)
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 4:
            return None
        numbers = [_finite(item) for item in value]
        if any(item is None for item in numbers):
            return None
        x, y, x_max, y_max = (float(item) for item in numbers if item is not None)
        if x_max > x and y_max > y:
            return (x, y, x_max - x, y_max - y)
    return None


def geometry_from_box(
    value: Any,
    *,
    source_width: int,
    source_height: int,
    crop: Any | None = None,
    frame_id: str | None = None,
    timestamp: float | None = None,
    precision: GeometryPrecision = GeometryPrecision.ESTIMATE,
    method: str = "model_estimate",
) -> Geometry | None:
    """把模型归一化框转换为源图像素坐标。"""

    box = normalized_box(value)
    if box is None:
        return None
    x, y, width, height = box

    if crop is not None:
        processed_x = crop.left + x * crop.width
        processed_y = crop.top + y * crop.height
        processed_w = width * crop.width
        processed_h = height * crop.height
        source_box = [
            processed_x / crop.scale_x,
            processed_y / crop.scale_y,
            (processed_x + processed_w) / crop.scale_x,
            (processed_y + processed_h) / crop.scale_y,
        ]
    else:
        source_box = [
            x * source_width,
            y * source_height,
            (x + width) * source_width,
            (y + height) * source_height,
        ]

    validated = pixel_bbox_validate(source_box, source_width, source_height, clamp=True)
    if validated is None:
        return None
    try:
        return Geometry(
            source_width=source_width,
            source_height=source_height,
            bbox=validated,
            precision=precision,
            method=method,
            frame_id=frame_id,
            timestamp=timestamp,
        )
    except (TypeError, ValueError):
        return None


def region_from_key(key: str) -> BodyRegion:
    mapping = {
        "lower_body": BodyRegion.LOWER_BODY,
        "left_leg": BodyRegion.LEFT_LEG,
        "right_leg": BodyRegion.RIGHT_LEG,
        "left_foot": BodyRegion.LEFT_FOOT,
        "right_foot": BodyRegion.RIGHT_FOOT,
    }
    return mapping.get(key, BodyRegion.LOWER_BODY)
