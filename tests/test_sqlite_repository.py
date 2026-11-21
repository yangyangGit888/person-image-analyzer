"""SQLite 仓库、任务领取、取消、恢复与生命周期测试。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.repositories.sqlite_repository import SQLiteRepository
from app.schemas.analysis_v2 import (
    AnalysisResultV2,
    ClothingItem,
    ConfidenceLevel,
    ItemCategory,
    Presence,
    Side,
    Visibility,
)
from app.schemas.mask import MaskPrecision, MaskRecord, MaskStatus


def _repo(tmp_path: Path) -> SQLiteRepository:
    return SQLiteRepository(tmp_path / "database" / "app.sqlite3")


def _analysis() -> AnalysisResultV2:
    return AnalysisResultV2(
        source_id="image-1",
        person_count=1,
        items=[
            ClothingItem(
                category=ItemCategory.HOSIERY,
                type="pantyhose",
                side=Side.LEFT,
                presence=Presence.DETECTED,
                visibility=Visibility.CLEAR,
                confidence=ConfidenceLevel.HIGH,
            )
        ],
    )


def test_analysis_and_item_roundtrip(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = _analysis()

    repo.save_analysis(result, source_path="uploads/a.jpg")
    loaded = repo.get_analysis(result.analysis_id)
    items = repo.list_items(result.analysis_id)

    assert loaded is not None
    assert loaded["source_path"] == "uploads/a.jpg"
    assert loaded["analysis_id"] == result.analysis_id
    assert len(items) == 1
    assert items[0]["category"] == "hosiery"
    assert repo.get_item(items[0]["item_id"])["side"] == "left"


def test_mask_is_linked_to_item_and_analysis(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = _analysis()
    repo.save_analysis(result)
    item_id = result.items[0].item_id
    mask = MaskRecord(
        item_id=item_id,
        analysis_id=result.analysis_id,
        source_id=result.source_id,
        path="masks/mask.png",
        width=100,
        height=100,
        precision=MaskPrecision.COARSE,
        status=MaskStatus.READY,
    )
    repo.save_mask(mask)

    loaded = repo.get_mask(mask.mask_id)
    assert loaded is not None
    assert loaded["path"] == "masks/mask.png"
    assert loaded["item_id"] == item_id


def test_claim_update_and_cancel_queued_task(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    created = repo.create_video_task(source_path="uploads/a.mp4")
    claimed = repo.claim_next_video_task("worker-1")

    assert claimed is not None
    assert claimed["task_id"] == created["task_id"]
    assert claimed["status"] == "preprocessing"
    assert claimed["attempts"] == 1

    updated = repo.update_video_task(
        created["task_id"],
        status="analyzing",
        stage="analyzing",
        progress=50,
    )
    assert updated is not None
    assert updated["progress"] == 50

    cancelled = repo.request_cancel(created["task_id"])
    assert cancelled is not None
    assert cancelled["cancel_requested"] is True
    assert cancelled["status"] == "analyzing"

    queued = repo.create_video_task(source_path="uploads/b.mp4")
    cancelled_queued = repo.request_cancel(queued["task_id"])
    assert cancelled_queued is not None
    assert cancelled_queued["status"] == "cancelled"


def test_completed_task_is_not_claimed_again(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    created = repo.create_video_task(source_path="uploads/a.mp4")
    repo.claim_next_video_task("worker-1")
    repo.update_video_task(
        created["task_id"],
        status="completed",
        stage="completed",
        progress=100,
    )

    assert repo.claim_next_video_task("worker-2") is None
    assert repo.count_video_tasks(status="completed") == 1


def test_stale_task_recovery_and_attempt_limit(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    created = repo.create_video_task(source_path="uploads/a.mp4")
    repo.claim_next_video_task("worker-1")
    old = datetime.now(UTC) - timedelta(minutes=10)
    repo.update_video_task(created["task_id"], heartbeat_at=old)

    recovered = repo.recover_stale_video_tasks(
        lease_seconds=60,
        max_attempts=2,
    )
    assert recovered == {"recovered": 1, "failed": 0}
    task = repo.get_video_task(created["task_id"])
    assert task is not None
    assert task["status"] == "queued"

    repo.claim_next_video_task("worker-2")
    repo.update_video_task(created["task_id"], heartbeat_at=old)
    failed = repo.recover_stale_video_tasks(lease_seconds=60, max_attempts=2)
    assert failed == {"recovered": 0, "failed": 1}
    task = repo.get_video_task(created["task_id"])
    assert task is not None
    assert task["status"] == "failed"


def test_delete_analysis_returns_related_paths(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = _analysis()
    repo.save_analysis(
        result,
        source_path="uploads/a.jpg",
        canonical_path="uploads/canonical.jpg",
    )
    repo.save_mask(
        MaskRecord(
            item_id=result.items[0].item_id,
            analysis_id=result.analysis_id,
            path="masks/mask.png",
            width=20,
            height=20,
        )
    )

    paths = repo.delete_analysis(result.analysis_id)
    assert "uploads/a.jpg" in paths
    assert "uploads/canonical.jpg" in paths
    assert "masks/mask.png" in paths
    assert repo.get_analysis(result.analysis_id) is None


def test_cleanup_expired_records(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    past = datetime.now(UTC) - timedelta(hours=1)
    result = _analysis()
    repo.save_analysis(result, source_path="uploads/a.jpg", expires_at=past)
    repo.save_mask(
        MaskRecord(
            item_id=result.items[0].item_id,
            analysis_id=result.analysis_id,
            path="masks/mask.png",
            width=20,
            height=20,
            expires_at=past,
        )
    )
    task = repo.create_video_task(source_path="uploads/a.mp4", expires_at=past)
    repo.update_video_task(
        task["task_id"],
        status="completed",
        stage="completed",
        progress=100,
    )

    cleaned = repo.cleanup_expired()
    assert cleaned["analyses"] >= 1
    assert cleaned["tasks"] >= 1
    assert "uploads/a.jpg" in cleaned["paths"]
