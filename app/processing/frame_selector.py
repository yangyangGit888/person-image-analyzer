"""视频候选帧质量过滤、相似帧去重和关键帧选择。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.core.config import Settings
from app.processing.image_quality import (
    FrameMetrics,
    analyze_frame,
    compute_motion_scores,
    hamming_distance,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FrameSelection:
    selected: list[FrameMetrics] = field(default_factory=list)
    all_metrics: list[FrameMetrics] = field(default_factory=list)
    candidate_count: int = 0
    quality_filtered: int = 0
    duplicates: int = 0
    warnings: list[str] = field(default_factory=list)


def _deduplicate(
    candidates: list[FrameMetrics],
    *,
    hamming_threshold: int,
    limit: int,
) -> tuple[list[FrameMetrics], int]:
    selected: list[FrameMetrics] = []
    duplicates = 0
    for candidate in candidates:
        duplicate = any(
            hamming_distance(candidate.dhash, existing.dhash) <= hamming_threshold
            for existing in selected
        )
        if duplicate:
            duplicates += 1
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected, duplicates


def select_frames(
    frames: list,
    *,
    settings: Settings,
) -> FrameSelection:
    metrics: list[FrameMetrics] = []
    for frame in frames:
        item = analyze_frame(
            frame.path,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            settings=settings,
        )
        if item is not None:
            metrics.append(item)

    compute_motion_scores(metrics)
    acceptable = [item for item in metrics if item.acceptable]
    rejected = [item for item in metrics if not item.acceptable]

    warnings: list[str] = []
    if not acceptable:
        warnings.append("没有通过质量筛选的帧，退回清晰度最高的候选帧")
        acceptable = sorted(metrics, key=lambda item: item.quality_score, reverse=True)[
            : settings.video_keyframe_count
        ]

    # 先按时间分桶，每桶选择质量与运动综合得分最高的代表帧，保证覆盖全片。
    ordered = sorted(acceptable, key=lambda item: item.timestamp)
    limit = max(1, settings.video_keyframe_count)
    bins = max(1, min(limit, len(ordered)))
    span = max(1e-6, ordered[-1].timestamp - ordered[0].timestamp)
    bucket_best: dict[int, FrameMetrics] = {}
    for item in ordered:
        index = min(
            bins - 1,
            int((item.timestamp - ordered[0].timestamp) / span * bins),
        )
        current = bucket_best.get(index)
        score = item.quality_score + item.motion_score * 0.35
        current_score = (
            current.quality_score + current.motion_score * 0.35
            if current is not None
            else -1.0
        )
        if current is None or score > current_score:
            bucket_best[index] = item

    selected, duplicates = _deduplicate(
        sorted(bucket_best.values(), key=lambda item: item.timestamp),
        hamming_threshold=settings.frame_dedup_hamming,
        limit=limit,
    )

    # 补足首尾和高质量帧，避免去重后时间覆盖不足。
    selected_ids = {item.frame_id for item in selected}
    for endpoint in (ordered[0], ordered[-1]):
        if endpoint.frame_id in selected_ids:
            continue
        if len(selected) < limit:
            selected.append(endpoint)
            selected_ids.add(endpoint.frame_id)
            continue
        replaceable = [
            item
            for item in selected
            if item.frame_id not in {ordered[0].frame_id, ordered[-1].frame_id}
        ]
        if replaceable:
            victim = min(
                replaceable,
                key=lambda item: item.quality_score + item.motion_score * 0.35,
            )
            selected.remove(victim)
            selected_ids.discard(victim.frame_id)
            selected.append(endpoint)
            selected_ids.add(endpoint.frame_id)
    if len(selected) < limit:
        remaining = sorted(
            (item for item in acceptable if item.frame_id not in selected_ids),
            key=lambda item: item.quality_score + item.motion_score * 0.35,
            reverse=True,
        )
        for item in remaining:
            if len(selected) >= limit:
                break
            if any(
                hamming_distance(item.dhash, existing.dhash)
                <= settings.frame_dedup_hamming
                for existing in selected
            ):
                duplicates += 1
                continue
            selected.append(item)
            selected_ids.add(item.frame_id)

    selected.sort(key=lambda item: item.timestamp)
    if rejected:
        warnings.append(f"过滤低质量帧 {len(rejected)} 张")
    if duplicates:
        warnings.append(f"去重相似帧 {duplicates} 张")
    if len(selected) < min(limit, len(acceptable)):
        warnings.append("可用关键帧数量少于目标数量")

    logger.info(
        "关键帧选择完成 candidates=%s selected=%s rejected=%s duplicates=%s",
        len(metrics),
        len(selected),
        len(rejected),
        duplicates,
    )
    return FrameSelection(
        selected=selected,
        all_metrics=metrics,
        candidate_count=len(frames),
        quality_filtered=len(rejected),
        duplicates=duplicates,
        warnings=warnings,
    )
