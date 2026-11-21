"""本地视频 Worker 生命周期测试（Mock 视频处理和模型）。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from PIL import Image, ImageDraw

from app.core.config import Settings
from app.processing.video_processor import ExtractedFrames, FrameRecord
from app.repositories.sqlite_repository import SQLiteRepository
from app.schemas.analysis import AnalysisData, ImageInfo
from app.schemas.analysis_v2 import (
    AnalysisResultV2,
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
from app.schemas.video import VideoInfo
from app.services.image_analysis_service import ImageAnalysisOutcome
from app.services.image_service import ImageService
from app.services.storage_service import StorageService
from app.workers.video_worker import VideoTaskWorker


class FakeProcessor:
    async def extract_frames(self, source, output_dir, *, info):
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for index in range(3):
            image = Image.new("RGB", (320, 240), "white")
            draw = ImageDraw.Draw(image)
            for offset in range(0, 320, 20):
                draw.line((offset, 0, 320 - offset, 240), fill="black", width=3)
            path = output_dir / f"frame_{index + 1:06d}.jpg"
            image.save(path, format="JPEG", quality=92)
            frames.append(
                FrameRecord(
                    frame_id=f"frame_{index + 1:06d}",
                    path=path,
                    timestamp=index * 0.5,
                )
            )
        return ExtractedFrames(frames=frames, sample_fps=2)


class FakeImageAnalysis:
    def __init__(self) -> None:
        self.calls = 0

    async def analyze(self, processed, **kwargs):
        self.calls += 1
        item = ClothingItem(
            category=ItemCategory.SHOES,
            type="sneakers",
            side=Side.LEFT,
            image_side=Side.LEFT,
            presence=Presence.DETECTED,
            visibility=Visibility.CLEAR,
            confidence=ConfidenceLevel.HIGH,
            geometry=Geometry(
                source_width=processed.source_width,
                source_height=processed.source_height,
                bbox=[20, 150, 120, 230],
            ),
        )
        analysis = AnalysisResultV2(
            source_type=SourceType.IMAGE,
            person_count=1,
            items=[item],
            feet=[
                FootState(
                    side=Side.LEFT,
                    image_side=Side.LEFT,
                    barefoot="not_barefoot",
                    visibility=Visibility.CLEAR,
                    confidence=ConfidenceLevel.HIGH,
                ),
                FootState(
                    side=Side.RIGHT,
                    image_side=Side.RIGHT,
                    barefoot="not_barefoot",
                    visibility=Visibility.CLEAR,
                    confidence=ConfidenceLevel.HIGH,
                ),
            ],
            usage={"total_tokens": 100},
        )
        data = AnalysisData(
            image_info=ImageInfo(
                width=processed.source_width,
                height=processed.source_height,
                person_count=1,
            ),
            persons=[],
        )
        return ImageAnalysisOutcome(
            data=data,
            analysis=analysis,
            calls=3,
            usage={"total_tokens": 100},
        )


def _build_worker(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        storage_root=str(tmp_path),
        video_max_ai_calls=9,
        video_keyframe_count=3,
        frame_min_sharpness=0,
        worker_task_timeout_seconds=60,
    )
    storage = StorageService(settings)
    repository = SQLiteRepository(settings.database_path)
    source = storage.path("uploads/videos/test.mp4")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"fake-video")
    task = repository.create_video_task(
        task_id="worker-task",
        source_path="uploads/videos/test.mp4",
        video_info=VideoInfo(
            duration_seconds=1.5,
            width=320,
            height=240,
            fps=30,
            codec="h264",
        ).model_dump(mode="json"),
    )
    fake_analysis = FakeImageAnalysis()
    worker = VideoTaskWorker(
        settings,
        storage,
        repository,
        FakeProcessor(),
        ImageService(settings),
        fake_analysis,
    )
    return worker, repository, fake_analysis, task


def test_worker_completes_task_and_persists_fused_result(tmp_path: Path) -> None:
    worker, repository, fake_analysis, _ = _build_worker(tmp_path)

    assert asyncio.run(worker.process_next()) is True
    task = repository.get_video_task("worker-task")
    assert task is not None
    assert task["status"] == "completed"
    assert task["progress"] == 100
    assert task["result_analysis_id"]
    analysis = repository.get_analysis(task["result_analysis_id"])
    assert analysis is not None
    assert analysis["source_type"] == "video"
    assert analysis["items"]
    assert fake_analysis.calls >= 1
    assert asyncio.run(worker.process_next()) is False


def test_worker_cancellation_stops_before_processing(tmp_path: Path) -> None:
    worker, repository, _, task = _build_worker(tmp_path)
    repository.update_video_task(task["task_id"], cancel_requested=True)

    asyncio.run(worker.process_task(task))
    updated = repository.get_video_task(task["task_id"])
    assert updated is not None
    assert updated["status"] == "cancelled"
