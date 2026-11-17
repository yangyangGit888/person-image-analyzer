"""分割模型统一接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class SegmentationPrompt:
    prompt_id: str
    box: tuple[int, int, int, int] | None = None
    positive_points: tuple[Point, ...] = field(default_factory=tuple)
    negative_points: tuple[Point, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class SegmentationMask:
    prompt_id: str
    mask: np.ndarray
    score: float | None = None
    method: str = "unknown"


class BaseSegmentationProvider(ABC):
    provider_name = "unknown"
    device = "cpu"
    supports_points = True
    supports_boxes = True

    @abstractmethod
    async def segment(
        self,
        image: np.ndarray,
        prompts: list[SegmentationPrompt],
    ) -> list[SegmentationMask]:
        """对同一张图片执行一组框/点提示分割。"""

    async def aclose(self) -> None:
        return None
