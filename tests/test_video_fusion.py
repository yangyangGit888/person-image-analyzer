"""视频多帧融合规则测试。"""

from __future__ import annotations

from app.fusion.video_result_fusion import FrameAnalysis, fuse_frame_results
from app.schemas.analysis_v2 import (
    AnalysisResultV2,
    BarefootStatus,
    ClothingItem,
    ConfidenceLevel,
    FootState,
    Geometry,
    ItemCategory,
    Presence,
    Side,
    SourceType,
    Visibility,
)


def _item(
    *,
    presence: Presence,
    visibility: Visibility,
    category: ItemCategory = ItemCategory.SHOES,
    side: Side = Side.LEFT,
    item_type: str = "sneakers",
) -> ClothingItem:
    return ClothingItem(
        category=category,
        type=item_type,
        side=side,
        image_side=side,
        presence=presence,
        visibility=visibility,
        confidence=ConfidenceLevel.HIGH,
        geometry=Geometry(
            source_width=640,
            source_height=480,
            bbox=[20, 300, 180, 440],
        ),
        evidence=[
            {
                "observation": "帧观察",
                "confidence": ConfidenceLevel.HIGH,
                "source": "frame",
            }
        ],
    )


def _frame(
    frame_id: str,
    timestamp: float,
    item: ClothingItem,
    *,
    barefoot: BarefootStatus = BarefootStatus.NOT_BAREFOOT,
    quality: float = 0.8,
) -> FrameAnalysis:
    analysis = AnalysisResultV2(
        source_type=SourceType.VIDEO,
        source_id="task-1",
        person_count=1,
        items=[item],
        feet=[
            FootState(
                side=Side.LEFT,
                image_side=Side.LEFT,
                barefoot=barefoot,
                visibility=Visibility.CLEAR,
            )
        ],
        usage={"total_tokens": 100},
    )
    return FrameAnalysis(
        frame_id=frame_id,
        timestamp=timestamp,
        quality=quality,
        analysis=analysis,
    )


def test_any_clear_detection_survives_invisible_frames() -> None:
    detected = _frame(
        "frame_1",
        0.0,
        _item(presence=Presence.DETECTED, visibility=Visibility.CLEAR),
    )
    invisible = _frame(
        "frame_2",
        1.0,
        _item(presence=Presence.NOT_VISIBLE, visibility=Visibility.NOT_VISIBLE),
    )

    result = fuse_frame_results([detected, invisible], source_id="task-1")
    shoes = [item for item in result.analysis.items if item.category is ItemCategory.SHOES]
    assert len(shoes) == 1
    assert shoes[0].presence is Presence.DETECTED
    assert {evidence.frame_id for evidence in shoes[0].evidence} >= {
        "frame_1",
        "frame_2",
    }


def test_clear_absence_in_all_frames_is_not_detected() -> None:
    first = _frame(
        "frame_1",
        0.0,
        _item(presence=Presence.NOT_DETECTED, visibility=Visibility.CLEAR),
    )
    second = _frame(
        "frame_2",
        1.0,
        _item(presence=Presence.NOT_DETECTED, visibility=Visibility.CLEAR),
    )

    result = fuse_frame_results([first, second], source_id="task-1")
    shoes = [item for item in result.analysis.items if item.category is ItemCategory.SHOES]
    assert shoes[0].presence is Presence.NOT_DETECTED


def test_detection_and_clear_absence_are_recorded_as_conflict() -> None:
    detected = _frame(
        "frame_1",
        0.0,
        _item(presence=Presence.DETECTED, visibility=Visibility.CLEAR),
    )
    absent = _frame(
        "frame_2",
        1.0,
        _item(presence=Presence.NOT_DETECTED, visibility=Visibility.CLEAR),
    )

    result = fuse_frame_results([detected, absent], source_id="task-1")
    shoes = [item for item in result.analysis.items if item.category is ItemCategory.SHOES]
    assert shoes[0].presence is Presence.DETECTED
    assert any(
        warning.code == "VIDEO_FRAME_CONFLICT"
        for warning in result.analysis.warnings
    )
    assert result.conflicting_frames


def test_barefoot_evidence_is_fused_conservatively() -> None:
    barefoot = _frame(
        "frame_1",
        0.0,
        _item(presence=Presence.NOT_DETECTED, visibility=Visibility.CLEAR),
        barefoot=BarefootStatus.BAREFOOT,
    )
    covered = _frame(
        "frame_2",
        1.0,
        _item(presence=Presence.DETECTED, visibility=Visibility.CLEAR),
        barefoot=BarefootStatus.NOT_BAREFOOT,
    )

    result = fuse_frame_results([barefoot, covered], source_id="task-1")
    assert result.analysis.feet[0].barefoot is BarefootStatus.NOT_BAREFOOT


def test_fused_item_ids_are_stable() -> None:
    frames = [
        _frame(
            "frame_1",
            0.0,
            _item(presence=Presence.DETECTED, visibility=Visibility.CLEAR),
        )
    ]
    first = fuse_frame_results(frames, source_id="task-1")
    second = fuse_frame_results(frames, source_id="task-1")
    assert [item.item_id for item in first.analysis.items] == [
        item.item_id for item in second.analysis.items
    ]
