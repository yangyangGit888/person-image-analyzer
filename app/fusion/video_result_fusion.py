"""视频关键帧结果融合。

不做简单多数投票。每个观察按清晰度、可见性、置信等级和帧质量加权；存在性
采用“任意清晰帧检测到即保留”的规则，同时显式记录冲突帧。不可见帧不会把
已检测物品投票成不存在。
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from app.relations.wearing_relation_engine import build_wearing_relations
from app.schemas.analysis_v2 import (
    AnalysisResultV2,
    AnalysisStatus,
    AnalysisWarning,
    BarefootStatus,
    BodyObservability,
    BodyRegion,
    ClothingItem,
    ConfidenceLevel,
    Evidence,
    EvidenceSource,
    FootState,
    ItemCategory,
    MaskStatus,
    PersonOrientation,
    Presence,
    RegionObservability,
    Side,
    SideBasis,
    SourceType,
    Visibility,
    WearingRelation,
)


@dataclass(slots=True)
class FrameAnalysis:
    frame_id: str
    timestamp: float
    quality: float
    analysis: AnalysisResultV2


@dataclass(slots=True)
class FusionResult:
    analysis: AnalysisResultV2
    conflicting_frames: list[dict[str, object]] = field(default_factory=list)


def _confidence_rank(value: ConfidenceLevel) -> int:
    return {
        ConfidenceLevel.LOW: 1,
        ConfidenceLevel.MEDIUM: 2,
        ConfidenceLevel.HIGH: 3,
    }.get(value, 0)


def _visibility_rank(value: Visibility) -> int:
    return {
        Visibility.NOT_VISIBLE: 0,
        Visibility.OCCLUDED: 1,
        Visibility.PARTIAL: 2,
        Visibility.CLEAR: 3,
    }.get(value, 0)


def _observation_score(item: ClothingItem, frame: FrameAnalysis) -> float:
    return (
        _confidence_rank(item.confidence) * 2.0
        + _visibility_rank(item.visibility)
        + frame.quality * 2.0
        + (2.0 if item.presence is Presence.DETECTED else 0.0)
    )


def _best(observations: list[tuple[FrameAnalysis, ClothingItem]]) -> tuple[FrameAnalysis, ClothingItem]:
    return max(observations, key=lambda pair: _observation_score(pair[1], pair[0]))


def _merge_presence(
    observations: list[tuple[FrameAnalysis, ClothingItem]],
) -> tuple[Presence, bool]:
    detected = [
        pair
        for pair in observations
        if pair[1].presence is Presence.DETECTED
    ]
    clear_not_detected = [
        pair
        for pair in observations
        if pair[1].presence is Presence.NOT_DETECTED
        and pair[1].visibility in (Visibility.CLEAR, Visibility.PARTIAL)
    ]
    all_not_visible = all(
        pair[1].presence is Presence.NOT_VISIBLE
        or pair[1].visibility is Visibility.NOT_VISIBLE
        for pair in observations
    )
    if detected:
        return Presence.DETECTED, bool(clear_not_detected)
    if all_not_visible:
        return Presence.NOT_VISIBLE, False
    if clear_not_detected and len(clear_not_detected) == len(observations):
        return Presence.NOT_DETECTED, False
    return Presence.UNCERTAIN, False


def _merge_evidence(
    observations: list[tuple[FrameAnalysis, ClothingItem]],
    *,
    limit: int = 12,
) -> list[Evidence]:
    evidence: list[Evidence] = []
    seen: set[tuple[str | None, float | None]] = set()
    for frame, item in observations:
        for entry in item.evidence[:2]:
            key = (frame.frame_id, frame.timestamp)
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                entry.model_copy(
                    update={
                        "frame_id": frame.frame_id,
                        "timestamp": frame.timestamp,
                        "source": EvidenceSource.FRAME,
                    }
                )
            )
            if len(evidence) >= limit:
                return evidence
    return evidence


def _fused_item_id(
    source_id: str,
    category: ItemCategory,
    side: Side,
    image_side: Side,
) -> str:
    raw = f"{source_id}:{category.value}:{side.value}:{image_side.value}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _fuse_items(
    frames: list[FrameAnalysis],
    *,
    source_id: str,
) -> tuple[list[ClothingItem], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[tuple[FrameAnalysis, ClothingItem]]] = defaultdict(list)
    for frame in frames:
        for item in frame.analysis.items:
            effective_side = (
                item.side if item.side is not Side.UNKNOWN else item.image_side
            )
            grouped[(item.category.value, effective_side.value)].append((frame, item))

    fused: list[ClothingItem] = []
    conflicts: list[dict[str, object]] = []
    for (_, _), observations in grouped.items():
        _, best_item = _best(observations)
        presence, conflict = _merge_presence(observations)
        if conflict:
            conflicts.append(
                {
                    "category": best_item.category.value,
                    "side": best_item.side.value,
                    "frames": [
                        {
                            "frame_id": frame.frame_id,
                            "timestamp": frame.timestamp,
                            "presence": item.presence.value,
                            "visibility": item.visibility.value,
                        }
                        for frame, item in observations
                    ],
                }
            )
        geometry = best_item.geometry
        item = best_item.model_copy(
            update={
                "item_id": _fused_item_id(
                    source_id,
                    best_item.category,
                    best_item.side,
                    best_item.image_side,
                ),
                "source_id": source_id,
                "presence": presence,
                "visibility": max(
                    (pair[1].visibility for pair in observations),
                    key=_visibility_rank,
                ),
                "confidence": max(
                    (pair[1].confidence for pair in observations),
                    key=_confidence_rank,
                ),
                "evidence": _merge_evidence(observations),
                "geometry": geometry,
                "occlusion_reason": list(
                    dict.fromkeys(
                        reason
                        for _, observed in observations
                        for reason in observed.occlusion_reason
                    )
                ),
                "mask_status": MaskStatus.NOT_REQUESTED,
                "mask_reference": None,
            }
        )
        fused.append(item)
    return fused, conflicts


def _merge_region(
    region: BodyRegion,
    frames: list[FrameAnalysis],
) -> RegionObservability:
    regions = []
    for frame in frames:
        observed = getattr(frame.analysis.body_observability, region.value)
        regions.append((frame, observed))
    best_visibility = max(
        (observed.visibility for _, observed in regions),
        key=_visibility_rank,
    )
    evidence = [
        Evidence(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            observation=f"{region.value} 可观察性来自关键帧",
            confidence=observed.confidence,
            source=EvidenceSource.FRAME,
        )
        for frame, observed in regions[:8]
    ]
    return RegionObservability(
        region=region,
        visibility=best_visibility,
        occlusion_reason=list(
            dict.fromkeys(
                reason
                for _, observed in regions
                for reason in observed.occlusion_reason
            )
        ),
        confidence=max(
            (observed.confidence for _, observed in regions),
            key=_confidence_rank,
        ),
        evidence=evidence,
    )


def _merge_body(frames: list[FrameAnalysis]) -> BodyObservability:
    return BodyObservability(
        lower_body=_merge_region(BodyRegion.LOWER_BODY, frames),
        left_leg=_merge_region(BodyRegion.LEFT_LEG, frames),
        right_leg=_merge_region(BodyRegion.RIGHT_LEG, frames),
        left_foot=_merge_region(BodyRegion.LEFT_FOOT, frames),
        right_foot=_merge_region(BodyRegion.RIGHT_FOOT, frames),
    )


def _fuse_feet(
    frames: list[FrameAnalysis],
    fused_items: list[ClothingItem],
) -> list[FootState]:
    grouped: dict[tuple[str, str], list[tuple[FrameAnalysis, FootState]]] = defaultdict(list)
    for frame in frames:
        for foot in frame.analysis.feet:
            effective = foot.side if foot.side is not Side.UNKNOWN else foot.image_side
            grouped[(foot.side.value, effective.value)].append((frame, foot))

    result: list[FootState] = []
    for (_, _), observations in grouped.items():
        _, best = max(observations, key=lambda pair: pair[0].quality)
        statuses = {foot.barefoot for _, foot in observations}
        if BarefootStatus.NOT_BAREFOOT in statuses:
            barefoot = BarefootStatus.NOT_BAREFOOT
        elif BarefootStatus.BAREFOOT in statuses:
            barefoot = BarefootStatus.BAREFOOT
        elif statuses == {BarefootStatus.NOT_VISIBLE}:
            barefoot = BarefootStatus.NOT_VISIBLE
        else:
            barefoot = BarefootStatus.UNKNOWN

        item_ids = [
            item.item_id
            for item in fused_items
            if item.side is best.side
            and item.category.value in {"hosiery", "socks", "shoes"}
            and item.presence is Presence.DETECTED
        ]
        evidence = [
            Evidence(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                observation=f"{best.side.value} 脚状态来自关键帧",
                confidence=foot.confidence,
                source=EvidenceSource.FRAME,
            )
            for frame, foot in observations[:8]
        ]
        result.append(
            FootState(
                side=best.side,
                image_side=best.image_side,
                barefoot=barefoot,
                visibility=max(
                    (foot.visibility for _, foot in observations),
                    key=_visibility_rank,
                ),
                item_ids=list(dict.fromkeys(item_ids)),
                evidence=evidence,
                uncertainty_reason=(
                    "多帧脚部状态不一致"
                    if len(
                        {
                            foot.barefoot
                            for _, foot in observations
                            if foot.barefoot is not BarefootStatus.UNKNOWN
                        }
                    )
                    > 1
                    else best.uncertainty_reason
                ),
            )
        )
    return result


def _merge_orientation(frames: list[FrameAnalysis]) -> tuple[PersonOrientation, list[str]]:
    known = [
        frame.analysis.person_orientation
        for frame in frames
        if frame.analysis.person_orientation is not PersonOrientation.UNKNOWN
    ]
    if not known:
        return PersonOrientation.UNKNOWN, []
    counts = Counter(known)
    orientation, count = counts.most_common(1)[0]
    if count < len(known):
        return PersonOrientation.UNKNOWN, ["多个关键帧的人物朝向不一致"]
    return orientation, []


def fuse_frame_results(
    frames: list[FrameAnalysis],
    *,
    source_id: str,
) -> FusionResult:
    if not frames:
        raise ValueError("没有可融合的关键帧结果")

    fused_items, conflicts = _fuse_items(frames, source_id=source_id)
    body = _merge_body(frames)
    feet = _fuse_feet(frames, fused_items)
    relations: list[WearingRelation] = build_wearing_relations(fused_items)
    orientation, orientation_warnings = _merge_orientation(frames)
    warnings: list[AnalysisWarning] = []
    seen_warnings: set[tuple[str, str]] = set()
    for frame in frames:
        for warning in frame.analysis.warnings:
            key = (warning.code, warning.message)
            if key in seen_warnings:
                continue
            seen_warnings.add(key)
            warnings.append(warning)
    for message in orientation_warnings:
        warnings.append(
            AnalysisWarning(code="VIDEO_ORIENTATION_CONFLICT", message=message)
        )
    for conflict in conflicts:
        warnings.append(
            AnalysisWarning(
                code="VIDEO_FRAME_CONFLICT",
                message="不同关键帧对同一物品的存在性判断不一致",
                detail=str(conflict),
            )
        )

    usage: dict[str, int] = {}
    prompt_versions: dict[str, str] = {}
    model: str | None = None
    for frame in frames:
        model = model or frame.analysis.model
        prompt_versions.update(frame.analysis.prompt_versions)
        for key, value in frame.analysis.usage.items():
            usage[key] = usage.get(key, 0) + value

    side_unknown = any(item.side is Side.UNKNOWN for item in fused_items)
    analysis = AnalysisResultV2(
        source_id=source_id,
        source_type=SourceType.VIDEO,
        status=AnalysisStatus.COMPLETED,
        person_count=1,
        person_orientation=orientation,
        side_basis=(
            frames[0].analysis.side_basis if frames else SideBasis.UNKNOWN
        ),
        side_assignment_confidence=(
            ConfidenceLevel.LOW
            if side_unknown
            else max(
                (frame.analysis.side_assignment_confidence for frame in frames),
                key=_confidence_rank,
            )
        ),
        items=fused_items,
        body_observability=body,
        feet=feet,
        wearing_relations=relations,
        evidence=[
            Evidence(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                observation="关键帧分析",
                confidence=ConfidenceLevel.MEDIUM,
                source=EvidenceSource.FRAME,
            )
            for frame in frames
        ],
        warnings=warnings,
        prompt_versions=prompt_versions,
        model=model,
        usage=usage,
    )
    return FusionResult(analysis=analysis, conflicting_frames=conflicts)
