"""MaskService、Placement Mask 与 PromptBuilder 单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import Settings
from app.core.exceptions import EditValidationError
from app.schemas.edit import EditOperation, MaskStrategy, PixelBox, RegionType
from app.services.mask_service import MaskService
from app.services.prompt_builder import PromptBuilder
from app.services.storage_service import StorageService


def _settings(tmp_path) -> Settings:
    return Settings(_env_file=None, storage_root=str(tmp_path))


def test_placement_hat_mask_is_valid(tmp_path) -> None:
    service = MaskService(StorageService(_settings(tmp_path)))
    mask = service.placement_mask(
        region_type=RegionType.HAT,
        person_bbox=PixelBox(x=200, y=120, width=300, height=700),
        image_width=800,
        image_height=1000,
    )
    quality = service.validate(
        mask,
        image_width=800,
        image_height=1000,
        region_type=RegionType.HAT,
        strategy=MaskStrategy.PLACEMENT,
    )

    assert mask.shape == (1000, 800)
    assert mask.any()
    assert quality.valid is True
    assert quality.area_ratio < 0.25


def test_placement_preserves_face_for_hat(tmp_path) -> None:
    service = MaskService(StorageService(_settings(tmp_path)))
    mask = service.placement_mask(
        region_type=RegionType.HAT,
        person_bbox=PixelBox(x=200, y=120, width=300, height=700),
        image_width=800,
        image_height=1000,
    )
    # 面部中心不应被帽子放置 Mask 完全覆盖
    assert not mask[260:340, 300:400].all()


def test_mask_validation_detects_size_mismatch(tmp_path) -> None:
    service = MaskService(StorageService(_settings(tmp_path)))
    mask = np.ones((100, 100), dtype=bool)
    quality = service.validate(
        mask,
        image_width=200,
        image_height=200,
        region_type=RegionType.HAT,
        strategy=MaskStrategy.PLACEMENT,
    )
    assert quality.valid is False
    assert quality.reason


def test_binary_png_roundtrip(tmp_path) -> None:
    service = MaskService(StorageService(_settings(tmp_path)))
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 10:20] = True
    data = service.mask_to_png_bytes(mask)
    restored = service.png_bytes_to_mask(data)

    assert restored.shape == mask.shape
    assert np.array_equal(restored, mask)
    # 严格二值 PNG，不含灰色边缘
    assert set(np.unique(np.asarray(service.png_bytes_to_mask(data)))) <= {False, True}


def test_dilate_erode_and_clip() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[4:6, 4:6] = True
    assert MaskService.dilate(mask, 1).sum() > mask.sum()
    assert MaskService.erode(mask, 1).sum() < mask.sum()
    container = np.zeros((10, 10), dtype=bool)
    container[0:5, 0:5] = True
    clipped = MaskService.clip(mask, container, expand=0)
    assert clipped.sum() <= mask.sum()


def test_prompt_builder_add_and_replace(tmp_path) -> None:
    builder = PromptBuilder(_settings(tmp_path))
    add = builder.build(
        person_ordinal=1,
        region_type=RegionType.HAT,
        operation=EditOperation.ADD,
        style_id="baseball_cap",
        color_id="black",
    )
    replace = builder.build(
        person_ordinal=2,
        region_type=RegionType.HAT,
        operation=EditOperation.REPLACE,
        style_id="beret",
        color_id="red",
    )

    assert "第1个人物" in add
    assert "添加" in add
    assert "棒球帽" in add
    assert "第2个人物" in replace
    assert "替换" in replace
    assert "贝雷帽" in replace
    assert "保持" in add


def test_prompt_builder_rejects_unknown_option(tmp_path) -> None:
    builder = PromptBuilder(_settings(tmp_path))
    with pytest.raises(EditValidationError):
        builder.build(
            person_ordinal=1,
            region_type=RegionType.HAT,
            operation=EditOperation.ADD,
            style_id="not-in-catalog",
            color_id="black",
        )
