"""图像编辑 Provider 工厂。"""

from __future__ import annotations

from app.ai.image_edit.base import BaseImageEditProvider
from app.ai.image_edit.mock_provider import MockImageEditProvider
from app.ai.image_edit.wanx_provider import WanxImageEditProvider
from app.core.config import Settings
from app.core.exceptions import ImageEditConfigError


def create_image_edit_provider(settings: Settings) -> BaseImageEditProvider:
    if settings.image_edit_provider == "mock":
        return MockImageEditProvider()
    if settings.image_edit_provider == "wanx":
        return WanxImageEditProvider(settings)
    raise ImageEditConfigError(
        detail=f"不支持的图像编辑 Provider：{settings.image_edit_provider}",
    )
