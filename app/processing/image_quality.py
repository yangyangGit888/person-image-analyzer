"""视频帧质量指标与感知哈希。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings


@dataclass(slots=True)
class FrameMetrics:
    frame_id: str
    path: Path
    timestamp: float
    width: int
    height: int
    sharpness: float
    brightness: float
    contrast: float
    edge_density: float
    dhash: int
    small_gray: bytes
    motion_score: float = 0.0
    quality_score: float = 0.0
    acceptable: bool = True
    reject_reason: str | None = None


def _dhash(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (resized[:, 1:] > resized[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def analyze_frame(
    path: Path,
    *,
    frame_id: str,
    timestamp: float,
    settings: Settings,
) -> FrameMetrics | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    max_side = 640
    if max(height, width) > max_side:
        ratio = max_side / max(height, width)
        image = cv2.resize(
            image,
            (max(1, round(width * ratio)), max(1, round(height * ratio))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

    acceptable = True
    reason = None
    if sharpness < settings.frame_min_sharpness:
        acceptable = False
        reason = "blurry"
    elif brightness < settings.frame_min_brightness:
        acceptable = False
        reason = "too_dark"
    elif brightness > settings.frame_max_brightness:
        acceptable = False
        reason = "too_bright"

    quality = (
        min(1.0, sharpness / max(1.0, settings.frame_min_sharpness * 4))
        * 0.5
        + min(1.0, contrast / 80.0) * 0.25
        + min(1.0, edge_density / 0.18) * 0.25
    )
    return FrameMetrics(
        frame_id=frame_id,
        path=path,
        timestamp=timestamp,
        width=width,
        height=height,
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
        edge_density=edge_density,
        dhash=_dhash(gray),
        small_gray=small.tobytes(),
        quality_score=quality,
        acceptable=acceptable,
        reject_reason=reason,
    )


def compute_motion_scores(metrics: list[FrameMetrics]) -> None:
    previous: np.ndarray | None = None
    for item in metrics:
        current = np.frombuffer(item.small_gray, dtype=np.uint8)
        if previous is not None and previous.shape == current.shape:
            item.motion_score = float(
                np.mean(np.abs(current.astype(np.int16) - previous.astype(np.int16)))
                / 255.0
            )
        previous = current
