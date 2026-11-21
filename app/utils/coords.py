"""Bounding Box 与坐标转换工具。

统一约定：

- 输出 BBox 为 ``[x_min, y_min, x_max, y_max]``。
- 原图左上角为原点，X 向右，Y 向下。
- 模型返回的坐标默认是相对当前输入图片的归一化坐标。
- 裁剪图的坐标必须通过裁剪偏移和尺寸映射回原图。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

BBox = list[int]


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pixel_bbox_validate(
    bbox: Sequence[float] | None,
    width: int,
    height: int,
    *,
    clamp: bool = True,
    min_size: int = 1,
) -> BBox | None:
    """校验像素 BBox；无法修复时返回 ``None``。"""

    if bbox is None or len(bbox) != 4 or width <= 0 or height <= 0:
        return None
    if not all(_is_finite_number(item) for item in bbox):
        return None

    x_min, y_min, x_max, y_max = (float(item) for item in bbox)
    if clamp:
        x_min = _clamp(x_min, 0.0, float(width))
        y_min = _clamp(y_min, 0.0, float(height))
        x_max = _clamp(x_max, 0.0, float(width))
        y_max = _clamp(y_max, 0.0, float(height))
    elif x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        return None

    left = max(0, min(width, int(round(x_min))))
    top = max(0, min(height, int(round(y_min))))
    right = max(0, min(width, int(round(x_max))))
    bottom = max(0, min(height, int(round(y_max))))

    if right - left < min_size or bottom - top < min_size:
        return None
    return [left, top, right, bottom]


def normalized_xyxy_to_pixels(
    bbox: Sequence[float] | None,
    width: int,
    height: int,
    *,
    clamp: bool = True,
) -> BBox | None:
    """把归一化 ``[x_min, y_min, x_max, y_max]`` 转为像素坐标。"""

    if bbox is None or len(bbox) != 4 or width <= 0 or height <= 0:
        return None
    if not all(_is_finite_number(item) for item in bbox):
        return None

    x_min, y_min, x_max, y_max = (float(item) for item in bbox)
    if clamp:
        x_min = _clamp(x_min, 0.0, 1.0)
        y_min = _clamp(y_min, 0.0, 1.0)
        x_max = _clamp(x_max, 0.0, 1.0)
        y_max = _clamp(y_max, 0.0, 1.0)
    elif not all(0.0 <= item <= 1.0 for item in (x_min, y_min, x_max, y_max)):
        return None

    return pixel_bbox_validate(
        [x_min * width, y_min * height, x_max * width, y_max * height],
        width,
        height,
        clamp=clamp,
    )


def normalized_xywh_to_pixels(
    box: Mapping[str, object] | Sequence[float] | None,
    width: int,
    height: int,
    *,
    clamp: bool = True,
) -> BBox | None:
    """把归一化 ``x/y/width/height`` 转为像素 ``[x1,y1,x2,y2]``。"""

    if box is None:
        return None

    if isinstance(box, Mapping):
        values = [
            box.get("x"),
            box.get("y"),
            box.get("width", box.get("w")),
            box.get("height", box.get("h")),
        ]
    elif isinstance(box, Sequence) and len(box) == 4:
        values = list(box)
    else:
        return None

    if not all(_is_finite_number(item) for item in values):
        return None
    x, y, box_width, box_height = (float(item) for item in values)
    if box_width <= 0 or box_height <= 0:
        return None
    return normalized_xyxy_to_pixels(
        [x, y, x + box_width, y + box_height],
        width,
        height,
        clamp=clamp,
    )


def crop_bbox_to_source(
    bbox: Sequence[float] | None,
    *,
    crop_x: int,
    crop_y: int,
    crop_width: int,
    crop_height: int,
    source_width: int,
    source_height: int,
) -> BBox | None:
    """把裁剪图内的归一化 BBox 映射回原图像素坐标。"""

    if bbox is None or len(bbox) != 4:
        return None
    if crop_width <= 0 or crop_height <= 0:
        return None
    if not all(_is_finite_number(item) for item in bbox):
        return None

    x_min, y_min, x_max, y_max = (float(item) for item in bbox)
    source_bbox = [
        crop_x + x_min * crop_width,
        crop_y + y_min * crop_height,
        crop_x + x_max * crop_width,
        crop_y + y_max * crop_height,
    ]
    return pixel_bbox_validate(source_bbox, source_width, source_height, clamp=True)


def scale_bbox(
    bbox: Sequence[float] | None,
    *,
    from_width: int,
    from_height: int,
    to_width: int,
    to_height: int,
) -> BBox | None:
    """按比例缩放像素 BBox。"""

    if bbox is None or len(bbox) != 4:
        return None
    if from_width <= 0 or from_height <= 0 or to_width <= 0 or to_height <= 0:
        return None
    scale_x = to_width / from_width
    scale_y = to_height / from_height
    return pixel_bbox_validate(
        [
            bbox[0] * scale_x,
            bbox[1] * scale_y,
            bbox[2] * scale_x,
            bbox[3] * scale_y,
        ],
        to_width,
        to_height,
        clamp=True,
    )


def bbox_area_ratio(bbox: Sequence[float] | None, width: int, height: int) -> float:
    if bbox is None or len(bbox) != 4 or width <= 0 or height <= 0:
        return 0.0
    x_min, y_min, x_max, y_max = (float(item) for item in bbox)
    area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    return min(1.0, area / float(width * height))


def bbox_iou(first: Sequence[float] | None, second: Sequence[float] | None) -> float:
    """计算两个像素 BBox 的 IoU，用于视频证据融合。"""

    if first is None or second is None or len(first) != 4 or len(second) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = (float(item) for item in first)
    bx1, by1, bx2, by2 = (float(item) for item in second)
    inter_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_width * inter_height
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0
