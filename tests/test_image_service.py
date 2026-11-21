"""图片归一化尺寸、缩放比例与哈希测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.config import Settings
from app.services.image_service import ImageService
from tests.helpers import make_image_bytes


def _process(tmp_path: Path, raw: bytes, **overrides):
    settings = Settings(
        _env_file=None,
        storage_root=str(tmp_path),
        **overrides,
    )
    return asyncio.run(ImageService(settings).process(raw))


def test_processed_image_records_source_and_scaled_dimensions(
    tmp_path: Path,
) -> None:
    processed = _process(
        tmp_path,
        make_image_bytes("PNG", size=(3000, 1000)),
        image_max_dimension=2048,
    )

    assert processed.source_width == 3000
    assert processed.source_height == 1000
    assert processed.width == 3000
    assert processed.height == 1000
    assert processed.processed_width == 2048
    assert processed.processed_height == 683
    assert processed.scale_x < 1
    assert processed.scale_y < 1
    assert len(processed.sha256) == 64
    assert processed.data.startswith(b"\xff\xd8")


def test_gif_is_converted_to_static_jpeg(tmp_path: Path) -> None:
    processed = _process(tmp_path, make_image_bytes("GIF", size=(320, 200)))

    assert processed.source_format == "GIF"
    assert processed.mime_type == "image/jpeg"
    assert processed.processed_width == 320
    assert processed.frames_used == 1
