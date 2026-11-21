"""视频任务 API 与取消/结果状态测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.schemas.analysis_v2 import (
    AnalysisResultV2,
    ClothingItem,
    ConfidenceLevel,
    Geometry,
    ItemCategory,
    Presence,
    Side,
    SourceType,
    Visibility,
)


def test_invalid_video_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/v1/analyze-video",
        files={"file": ("bad.mp4", b"not-a-video", "video/mp4")},
    )

    assert response.status_code in {400, 415, 422}
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] in {
        "INVALID_VIDEO",
        "UNSUPPORTED_VIDEO_FORMAT",
        "VIDEO_DECODE_FAILED",
    }
    assert "test-key" not in response.text


def test_task_not_found(client: TestClient) -> None:
    response = client.get("/api/v1/video-tasks/missing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "RESOURCE_NOT_FOUND"


def test_queued_task_status_and_cancel(client: TestClient) -> None:
    repository = client.app.state.repository
    repository.create_video_task(task_id="task-queued")

    status = client.get("/api/v1/video-tasks/task-queued")
    assert status.status_code == 200
    assert status.json()["data"]["status"] == "queued"

    not_ready = client.get("/api/v1/video-tasks/task-queued/result")
    assert not_ready.status_code == 409
    assert not_ready.json()["error"]["code"] == "TASK_NOT_COMPLETED"

    cancelled = client.post("/api/v1/video-tasks/task-queued/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["data"]["status"] == "cancelled"

    result = client.get("/api/v1/video-tasks/task-queued/result")
    assert result.status_code == 409
    assert result.json()["error"]["code"] == "TASK_CANCELLED"


def test_completed_task_result_is_returned(client: TestClient) -> None:
    repository = client.app.state.repository
    task = repository.create_video_task(task_id="task-completed")
    analysis = AnalysisResultV2(
        source_id="task-completed",
        source_type=SourceType.VIDEO,
        person_count=1,
        items=[
            ClothingItem(
                category=ItemCategory.SHOES,
                type="sneakers",
                side=Side.LEFT,
                image_side=Side.LEFT,
                presence=Presence.DETECTED,
                visibility=Visibility.CLEAR,
                confidence=ConfidenceLevel.HIGH,
                geometry=Geometry(
                    source_width=640,
                    source_height=480,
                    bbox=[20, 300, 180, 440],
                    frame_id="frame_000001",
                    timestamp=0.0,
                ),
            )
        ],
        usage={"total_tokens": 100},
    )
    repository.save_analysis(analysis, source_path="uploads/videos/test.mp4")
    repository.update_video_task(
        task["task_id"],
        status="completed",
        stage="completed",
        progress=100,
        result_analysis_id=analysis.analysis_id,
    )

    response = client.get("/api/v1/video-tasks/task-completed/result")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["analysis_id"] == analysis.analysis_id
    assert data["source_type"] == "video"
    assert data["items"][0]["category"] == "shoes"
