"""分割 Provider 工厂。"""

from __future__ import annotations

from app.ai.segmentation.base import BaseSegmentationProvider
from app.ai.segmentation.mock_provider import (
    BBoxSegmentationProvider,
    MockSegmentationProvider,
)
from app.ai.segmentation.sam2_provider import SAM2SegmentationProvider
from app.core.config import Settings
from app.core.exceptions import AIConfigError


def create_segmentation_provider(settings: Settings) -> BaseSegmentationProvider:
    if settings.segmentation_provider == "sam2":
        return SAM2SegmentationProvider(settings)
    if settings.segmentation_provider == "mock":
        return MockSegmentationProvider()
    if settings.segmentation_provider == "bbox":
        return BBoxSegmentationProvider()
    raise AIConfigError(detail=f"不支持的分割 Provider：{settings.segmentation_provider}")
