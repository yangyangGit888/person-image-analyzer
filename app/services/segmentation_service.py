"""人物与目标区域的分割编排。

SAM2 只负责像素分割，不负责语义判断；语义由 DeepSeek/区域检测服务提供。
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from app.ai.segmentation.base import (
    BaseSegmentationProvider,
    SegmentationMask,
    SegmentationPrompt,
)
from app.core.exceptions import SegmentationError
from app.schemas.edit import PixelBox


class SegmentationService:
    def __init__(self, provider: BaseSegmentationProvider) -> None:
        self._provider = provider

    @staticmethod
    def image_from_bytes(data: bytes) -> np.ndarray:
        with Image.open(BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"))

    async def segment_persons(
        self,
        image: np.ndarray,
        boxes: list[PixelBox],
    ) -> list[SegmentationMask]:
        prompts = [
            SegmentationPrompt(prompt_id=f"person-{index}", box=box.as_xyxy())
            for index, box in enumerate(boxes)
        ]
        results = await self._provider.segment(image, prompts)
        by_id = {result.prompt_id: result for result in results}
        ordered: list[SegmentationMask] = []
        for index in range(len(boxes)):
            result = by_id.get(f"person-{index}")
            if result is None:
                raise SegmentationError(detail="人物 Mask 数量与人物数量不一致")
            ordered.append(result)
        return ordered

    async def segment_object(
        self,
        image: np.ndarray,
        box: PixelBox,
        *,
        prompt_id: str,
    ) -> SegmentationMask:
        height, width = image.shape[:2]
        area_ratio = (box.width * box.height) / float(width * height)
        use_crop = area_ratio < 0.08 or box.width < width * 0.12
        if not use_crop:
            results = await self._provider.segment(
                image,
                [SegmentationPrompt(prompt_id=prompt_id, box=box.as_xyxy())],
            )
            if not results:
                raise SegmentationError(detail="目标分割未返回结果")
            result = results[0]
            return self._ensure_full_size(result, width, height)

        crop_x1 = max(0, box.x - int(box.width * 0.35))
        crop_y1 = max(0, box.y - int(box.height * 0.35))
        crop_x2 = min(width, box.right + int(box.width * 0.35))
        crop_y2 = min(height, box.bottom + int(box.height * 0.35))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            raise SegmentationError(detail="无法裁剪目标区域")
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        local_box = (
            box.x - crop_x1,
            box.y - crop_y1,
            min(box.right - crop_x1, crop_x2 - crop_x1),
            min(box.bottom - crop_y1, crop_y2 - crop_y1),
        )
        results = await self._provider.segment(
            crop,
            [SegmentationPrompt(prompt_id=prompt_id, box=local_box)],
        )
        if not results:
            raise SegmentationError(detail="局部裁剪分割未返回结果")
        crop_mask = results[0].mask
        full = np.zeros((height, width), dtype=bool)
        target_h = min(crop_mask.shape[0], crop_y2 - crop_y1)
        target_w = min(crop_mask.shape[1], crop_x2 - crop_x1)
        full[crop_y1 : crop_y1 + target_h, crop_x1 : crop_x1 + target_w] = crop_mask[
            :target_h, :target_w
        ]
        return SegmentationMask(
            prompt_id=prompt_id,
            mask=full,
            score=results[0].score,
            method=f"{results[0].method}_crop",
        )

    @staticmethod
    def _ensure_full_size(
        result: SegmentationMask,
        width: int,
        height: int,
    ) -> SegmentationMask:
        if result.mask.shape != (height, width):
            raise SegmentationError(detail="分割 Mask 与原图尺寸不一致")
        return result
