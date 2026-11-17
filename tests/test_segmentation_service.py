"""分割编排与坐标映射测试。"""

from __future__ import annotations

import asyncio

import numpy as np

from app.ai.segmentation.mock_provider import MockSegmentationProvider
from app.schemas.edit import PixelBox
from app.services.segmentation_service import SegmentationService


def test_segment_multiple_persons() -> None:
    service = SegmentationService(MockSegmentationProvider())
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    results = asyncio.run(service.segment_persons(
        image,
        [
            PixelBox(x=10, y=10, width=80, height=150),
            PixelBox(x=180, y=20, width=90, height=160),
        ],
    ))
    assert len(results) == 2
    assert results[0].mask[20, 20]
    assert not results[0].mask[100, 200]
    assert results[1].mask[30, 200]


def test_small_object_crop_maps_back_to_full_image() -> None:
    service = SegmentationService(MockSegmentationProvider())
    image = np.zeros((400, 500, 3), dtype=np.uint8)
    box = PixelBox(x=300, y=100, width=30, height=20)
    result = asyncio.run(service.segment_object(image, box, prompt_id="earring"))

    assert result.mask.shape == (400, 500)
    assert result.mask[105:115, 305:320].any()
    assert "crop" in result.method
