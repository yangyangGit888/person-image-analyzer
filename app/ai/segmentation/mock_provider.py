"""测试与本地降级使用的矩形分割 Provider。"""

from __future__ import annotations

import numpy as np

from app.ai.segmentation.base import (
    BaseSegmentationProvider,
    SegmentationMask,
    SegmentationPrompt,
)


class MockSegmentationProvider(BaseSegmentationProvider):
    provider_name = "mock"
    device = "cpu"

    async def segment(
        self,
        image: np.ndarray,
        prompts: list[SegmentationPrompt],
    ) -> list[SegmentationMask]:
        height, width = image.shape[:2]
        results: list[SegmentationMask] = []
        for prompt in prompts:
            mask = np.zeros((height, width), dtype=bool)
            if prompt.box is not None:
                x1, y1, x2, y2 = prompt.box
                x1 = max(0, min(width, x1))
                y1 = max(0, min(height, y1))
                x2 = max(0, min(width, x2))
                y2 = max(0, min(height, y2))
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = True
            elif prompt.positive_points:
                x, y = prompt.positive_points[0]
                radius = max(4, min(width, height) // 40)
                left = max(0, int(x) - radius)
                right = min(width, int(x) + radius)
                top = max(0, int(y) - radius)
                bottom = min(height, int(y) + radius)
                mask[top:bottom, left:right] = True
            results.append(
                SegmentationMask(
                    prompt_id=prompt.prompt_id,
                    mask=mask,
                    score=0.9 if mask.any() else 0.0,
                    method="mock_rectangle",
                )
            )
        return results


class BBoxSegmentationProvider(BaseSegmentationProvider):
    """无 SAM2 时的几何降级：直接用框生成矩形 Mask，仅用于开发验证。"""

    provider_name = "bbox"
    device = "cpu"

    async def segment(
        self,
        image: np.ndarray,
        prompts: list[SegmentationPrompt],
    ) -> list[SegmentationMask]:
        provider = MockSegmentationProvider()
        return await provider.segment(image, prompts)
