"""BBox 与坐标转换测试。"""

from __future__ import annotations

from app.utils.coords import (
    bbox_area_ratio,
    bbox_iou,
    crop_bbox_to_source,
    normalized_xywh_to_pixels,
    normalized_xyxy_to_pixels,
    pixel_bbox_validate,
    scale_bbox,
)


def test_normalized_xyxy_to_pixels() -> None:
    assert normalized_xyxy_to_pixels([0.1, 0.2, 0.5, 0.6], 100, 200) == [
        10,
        40,
        50,
        120,
    ]


def test_normalized_xywh_to_pixels() -> None:
    assert normalized_xywh_to_pixels(
        {"x": 0.1, "y": 0.2, "width": 0.4, "height": 0.4},
        100,
        100,
    ) == [10, 20, 50, 60]


def test_invalid_and_clamped_bbox() -> None:
    assert pixel_bbox_validate([-10, -10, 120, 120], 100, 100) == [0, 0, 100, 100]
    assert pixel_bbox_validate([50, 50, 50, 80], 100, 100) is None
    assert pixel_bbox_validate([1, 2, 3], 100, 100) is None


def test_crop_bbox_maps_back_to_source() -> None:
    assert crop_bbox_to_source(
        [0.25, 0.25, 0.75, 0.75],
        crop_x=100,
        crop_y=50,
        crop_width=200,
        crop_height=100,
        source_width=500,
        source_height=300,
    ) == [150, 75, 250, 125]


def test_scale_bbox() -> None:
    assert scale_bbox(
        [10, 20, 30, 40],
        from_width=100,
        from_height=100,
        to_width=200,
        to_height=200,
    ) == [20, 40, 60, 80]


def test_bbox_metrics() -> None:
    assert bbox_area_ratio([0, 0, 50, 50], 100, 100) == 0.25
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
