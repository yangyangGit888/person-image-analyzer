"""图像编辑模型统一接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ImageEditRequest:
    image_data: bytes
    image_mime: str
    image_width: int
    image_height: int
    mask_data: bytes
    mask_mime: str
    mask_width: int
    mask_height: int
    prompt: str
    seed: int | None = None


@dataclass(slots=True)
class ImageEditTask:
    task_id: str
    status: str
    result_urls: list[str] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class BaseImageEditProvider(ABC):
    provider_name = "unknown"
    model = "unknown"

    @abstractmethod
    async def submit(self, request: ImageEditRequest) -> ImageEditTask:
        """提交异步编辑任务。"""

    @abstractmethod
    async def fetch(self, task_id: str) -> ImageEditTask:
        """查询任务状态。"""

    @abstractmethod
    async def download(self, url: str) -> bytes:
        """下载编辑结果图片。"""

    async def aclose(self) -> None:
        return None
