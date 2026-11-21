"""图片裁剪与坐标上下文。

裁剪始终从模型实际看到的规范化 JPEG 中执行，并保存裁剪偏移、尺寸和缩放
比例。模型返回的裁剪图归一化坐标必须先映射回处理图，再除以缩放比例映射
回原图像素。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from app.ai.base import ImageInput
from app.services.image_service import ProcessedImage
from app.utils.coords import pixel_bbox_validate


@dataclass(frozen=True, slots=True)
class CropRegion:
    name: str
    left: int
    top: int
    width: int
    height: int
    scale_x: float
    scale_y: float
    source_width: int
    source_height: int
    image: ImageInput

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


def _clamp_box(
    bbox: Sequence[int],
    width: int,
    height: int,
    *,
    padding_ratio: float,
) -> tuple[int, int, int, int] | None:
    validated = pixel_bbox_validate(bbox, width, height, clamp=True)
    if validated is None:
        return None
    x_min, y_min, x_max, y_max = validated
    box_width = x_max - x_min
    box_height = y_max - y_min
    pad_x = int(round(box_width * padding_ratio))
    pad_y = int(round(box_height * padding_ratio))
    return (
        max(0, x_min - pad_x),
        max(0, y_min - pad_y),
        min(width, x_max + pad_x),
        min(height, y_max + pad_y),
    )


def make_crop(
    processed: ProcessedImage,
    name: str,
    *,
    source_bbox: Sequence[int] | None,
    padding_ratio: float,
    min_side: int,
    max_side: int,
) -> CropRegion | None:
    """从处理图裁剪区域；``source_bbox`` 为原图像素坐标。"""

    scale_x = processed.scale_x or 1.0
    scale_y = processed.scale_y or 1.0

    if source_bbox is None:
        processed_box = [0, 0, processed.processed_width, processed.processed_height]
    else:
        processed_box = [
            int(round(source_bbox[0] * scale_x)),
            int(round(source_bbox[1] * scale_y)),
            int(round(source_bbox[2] * scale_x)),
            int(round(source_bbox[3] * scale_y)),
        ]

    clamped = _clamp_box(
        processed_box,
        processed.processed_width,
        processed.processed_height,
        padding_ratio=padding_ratio,
    )
    if clamped is None:
        return None
    left, top, right, bottom = clamped
    width = right - left
    height = bottom - top
    if width < 32 or height < 32:
        return None

    try:
        with Image.open(BytesIO(processed.data)) as source:
            source.load()
            crop = source.convert("RGB").crop((left, top, right, bottom))
        crop_width, crop_height = crop.size

        longest = max(crop_width, crop_height)
        target_scale = 1.0
        if longest < min_side:
            target_scale = min_side / longest
        elif longest > max_side:
            target_scale = max_side / longest
        if abs(target_scale - 1.0) > 0.01:
            crop = crop.resize(
                (
                    max(1, int(round(crop_width * target_scale))),
                    max(1, int(round(crop_height * target_scale))),
                ),
                Image.Resampling.LANCZOS,
            )

        output = BytesIO()
        crop.save(output, format="JPEG", quality=92, optimize=True, subsampling=0)
        data = output.getvalue()
        image = ImageInput(
            data=data,
            mime_type="image/jpeg",
            width=crop.size[0],
            height=crop.size[1],
            filename=f"{name}.jpg",
        )
    except OSError:
        return None

    return CropRegion(
        name=name,
        left=left,
        top=top,
        width=width,
        height=height,
        scale_x=scale_x,
        scale_y=scale_y,
        source_width=processed.source_width,
        source_height=processed.source_height,
        image=image,
    )
