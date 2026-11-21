"""SQLite + 本地进程视频 Worker。"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from app.ai.factory import create_vision_provider
from app.ai.prompt_manager import PromptManager
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    AppError,
    InvalidVideoError,
    NoValidFramesError,
    TaskCancelledError,
    TaskTimeoutError,
    UnsupportedMultiPersonError,
)
from app.fusion.video_result_fusion import FrameAnalysis, fuse_frame_results
from app.processing.frame_selector import FrameSelection, select_frames
from app.processing.video_processor import FrameRecord, VideoProcessor
from app.repositories.sqlite_repository import SQLiteRepository
from app.schemas.analysis_v2 import AnalysisStatus, AnalysisWarning
from app.schemas.video import VideoInfo
from app.services.analysis_service import AnalysisService
from app.services.image_analysis_service import ImageAnalysisService
from app.services.image_service import ImageService
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)


class WorkerStopping(Exception):
    pass


def _now() -> datetime:
    return datetime.now(UTC)


def _pick_evenly(items: list[Any], count: int) -> list[Any]:
    if count <= 1 or len(items) <= count:
        return items[: max(1, count)]
    if len(items) == 0:
        return []
    indices = {
        round(index * (len(items) - 1) / (count - 1))
        for index in range(count)
    }
    return [items[index] for index in sorted(indices)]


class VideoTaskWorker:
    def __init__(
        self,
        settings: Settings,
        storage: StorageService,
        repository: SQLiteRepository,
        processor: VideoProcessor,
        image_service: ImageService,
        image_analysis_service: ImageAnalysisService,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._repository = repository
        self._processor = processor
        self._image_service = image_service
        self._image_analysis = image_analysis_service
        self._stop = asyncio.Event()
        self.worker_id = (
            f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        )

    def stop(self) -> None:
        self._stop.set()

    async def _update(self, task_id: str, **fields: Any) -> dict[str, Any] | None:
        return await asyncio.to_thread(
            self._repository.update_video_task,
            task_id,
            **fields,
        )

    async def _task(self, task_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._repository.get_video_task, task_id)

    def _check_control(self, task: dict[str, Any], started: float) -> None:
        if self._stop.is_set():
            raise WorkerStopping()
        if time.monotonic() - started > self._settings.worker_task_timeout_seconds:
            raise TaskTimeoutError()
        if task.get("cancel_requested"):
            raise TaskCancelledError()

    async def _refresh_and_check(self, task_id: str, started: float) -> dict[str, Any]:
        task = await self._task(task_id)
        if task is None:
            raise TaskCancelledError(detail="任务在运行期间被删除")
        self._check_control(task, started)
        return task

    async def _annotate_frame(
        self,
        analysis: Any,
        frame: FrameRecord,
    ) -> None:
        for item in analysis.items:
            item.source_id = analysis.source_id
            for evidence in item.evidence:
                evidence.frame_id = frame.frame_id
                evidence.timestamp = frame.timestamp
                if evidence.bbox is not None:
                    evidence.bbox.frame_id = frame.frame_id
                    evidence.bbox.timestamp = frame.timestamp
            if item.geometry is not None:
                item.geometry.frame_id = frame.frame_id
                item.geometry.timestamp = frame.timestamp

    async def _analyze_frames(
        self,
        task: dict[str, Any],
        selection: FrameSelection,
        *,
        started: float,
    ) -> tuple[list[FrameAnalysis], dict[str, int], list[str], int]:
        max_calls = self._settings.video_max_ai_calls
        per_frame_budget = max(1, min(3, max_calls))
        max_initial = max(1, max_calls // per_frame_budget)
        candidates = _pick_evenly(selection.selected, max_initial)
        metrics = {item.frame_id: item for item in selection.all_metrics}
        analyzed_ids: set[str] = set()
        frame_results: list[FrameAnalysis] = []
        usage: dict[str, int] = {}
        warnings: list[str] = []
        used_calls = 0
        api_calls = 0

        async def analyze_one(frame: FrameRecord, index: int, total: int) -> None:
            nonlocal used_calls, api_calls
            await self._refresh_and_check(task["task_id"], started)
            raw = frame.path.read_bytes()
            processed = await self._image_service.process(raw)
            outcome = await self._image_analysis.analyze(
                processed,
                filename=frame.frame_id,
                request_id=task["task_id"],
                max_calls=per_frame_budget,
                include_clothing_detail=False,
            )
            if outcome.analysis.status is AnalysisStatus.UNSUPPORTED_MULTI_PERSON:
                raise UnsupportedMultiPersonError()
            await self._annotate_frame(outcome.analysis, frame)
            metric = metrics.get(frame.frame_id)
            frame_results.append(
                FrameAnalysis(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    quality=metric.quality_score if metric else 0.5,
                    analysis=outcome.analysis,
                )
            )
            analyzed_ids.add(frame.frame_id)
            used_calls += outcome.calls
            api_calls += outcome.calls
            for key, value in outcome.usage.items():
                usage[key] = usage.get(key, 0) + value
            progress = 40 + int(45 * (index + 1) / max(1, total))
            await self._update(
                task["task_id"],
                progress=min(85, progress),
                heartbeat_at=_now(),
                usage=usage,
            )

        for index, frame in enumerate(candidates):
            if used_calls + per_frame_budget > max_calls:
                warnings.append("达到单任务最大 AI 调用次数，停止补充帧")
                break
            await analyze_one(frame, index, len(candidates))

        # 若预算仍有剩余且还有关键帧未分析，补充最多 2 帧以提高脚部/遮挡覆盖。
        supplemental = 0
        remaining = [
            frame
            for frame in selection.selected
            if frame.frame_id not in analyzed_ids
        ]
        remaining.sort(
            key=lambda frame: (
                metrics.get(frame.frame_id).quality_score
                if metrics.get(frame.frame_id)
                else 0.0
            ),
            reverse=True,
        )
        for frame in remaining:
            if supplemental >= 2:
                break
            if used_calls + per_frame_budget > max_calls:
                break
            await analyze_one(frame, len(candidates) + supplemental, len(candidates) + 2)
            supplemental += 1

        if not frame_results:
            raise NoValidFramesError(detail="没有成功分析的关键帧")
        return frame_results, usage, warnings, api_calls

    async def _cleanup_frames(self, task_id: str) -> None:
        await asyncio.to_thread(self._storage.delete_tree, f"frames/{task_id}")

    def cleanup_expired_frames(self) -> int:
        cutoff = time.time() - self._settings.frame_retention_hours * 3600
        deleted = 0
        if not self._storage.frames.is_dir():
            return 0
        for child in self._storage.frames.iterdir():
            if not child.is_dir():
                continue
            try:
                if child.stat().st_mtime < cutoff:
                    self._storage.delete_tree(f"frames/{child.name}")
                    deleted += 1
            except OSError:
                continue
        return deleted

    async def process_task(self, task: dict[str, Any]) -> None:
        task_id = str(task["task_id"])
        started = time.monotonic()
        frames_dir = self._storage.frames / task_id
        try:
            if not task.get("source_path"):
                raise InvalidVideoError(detail="任务缺少源视频")
            source = self._storage.path(str(task["source_path"]))
            if not source.is_file():
                raise InvalidVideoError(detail="源视频不存在或已过期")
            info = VideoInfo.model_validate(task.get("video_info") or {})
            await self._update(
                task_id,
                status="preprocessing",
                stage="preprocessing",
                progress=5,
                worker_id=self.worker_id,
                started_at=_now(),
                heartbeat_at=_now(),
                error_code=None,
                error_detail=None,
            )
            await self._refresh_and_check(task_id, started)

            await asyncio.to_thread(self._storage.delete_tree, f"frames/{task_id}")
            extracted = await self._processor.extract_frames(
                source,
                frames_dir,
                info=info,
            )
            await self._update(
                task_id,
                status="selecting_frames",
                stage="selecting_frames",
                progress=25,
                heartbeat_at=_now(),
            )
            await self._refresh_and_check(task_id, started)
            selection = await asyncio.to_thread(
                select_frames,
                extracted.frames,
                settings=self._settings,
            )
            if not selection.selected:
                raise NoValidFramesError()
            await self._update(
                task_id,
                status="analyzing",
                stage="analyzing",
                progress=40,
                warnings=selection.warnings,
                heartbeat_at=_now(),
            )

            frame_results, usage, frame_warnings, api_calls = await self._analyze_frames(
                task,
                selection,
                started=started,
            )
            usage["api_calls"] = api_calls
            await self._refresh_and_check(task_id, started)
            await self._update(
                task_id,
                status="merging",
                stage="merging",
                progress=90,
                heartbeat_at=_now(),
            )
            fusion = await asyncio.to_thread(
                fuse_frame_results,
                frame_results,
                source_id=task_id,
            )
            analysis = fusion.analysis
            analysis.usage["api_calls"] = api_calls
            for message in frame_warnings:
                analysis.warnings.append(
                    AnalysisWarning(code="VIDEO_AI_BUDGET", message=message)
                )
            for message in selection.warnings:
                analysis.warnings.append(
                    AnalysisWarning(code="VIDEO_FRAME_SELECTION", message=message)
                )
            if any(
                frame.analysis.status is AnalysisStatus.PARTIAL
                for frame in frame_results
            ):
                analysis.status = AnalysisStatus.PARTIAL
            expires_at = _now() + timedelta(
                days=self._settings.result_retention_days
            )
            await asyncio.to_thread(
                self._repository.save_analysis,
                analysis,
                source_path=str(task["source_path"]),
                expires_at=expires_at,
            )
            await self._update(
                task_id,
                status="completed",
                stage="completed",
                progress=100,
                result_analysis_id=analysis.analysis_id,
                usage=usage,
                warnings=[warning.message for warning in analysis.warnings],
                finished_at=_now(),
                heartbeat_at=_now(),
                error_code=None,
                error_detail=None,
            )
            logger.info(
                "视频任务完成 task_id=%s frames=%s api_calls=%s calls_budget=%s "
                "duration_ms=%.1f usage=%s",
                task_id,
                len(frame_results),
                api_calls,
                self._settings.video_max_ai_calls,
                (time.monotonic() - started) * 1000,
                usage,
            )
        except WorkerStopping:
            await self._update(
                task_id,
                status="queued",
                stage="queued",
                worker_id=None,
                heartbeat_at=None,
                error_code="WORKER_STOPPING",
                error_detail="Worker 正在停止，任务重新排队",
            )
            logger.warning("Worker 停止，任务重新排队 task_id=%s", task_id)
        except TaskCancelledError:
            await self._update(
                task_id,
                status="cancelled",
                stage="cancelled",
                finished_at=_now(),
                heartbeat_at=_now(),
                error_code="TASK_CANCELLED",
                error_detail="用户取消任务",
            )
            await self._cleanup_frames(task_id)
            logger.info("视频任务取消 task_id=%s", task_id)
        except AppError as exc:
            await self._update(
                task_id,
                status="failed",
                stage="failed",
                finished_at=_now(),
                heartbeat_at=_now(),
                error_code=exc.code,
                error_detail=exc.detail,
            )
            logger.warning(
                "视频任务失败 task_id=%s code=%s detail=%s",
                task_id,
                exc.code,
                exc.detail,
            )
        except Exception:  # noqa: BLE001
            await self._update(
                task_id,
                status="failed",
                stage="failed",
                finished_at=_now(),
                heartbeat_at=_now(),
                error_code="INTERNAL_ERROR",
                error_detail="视频任务处理异常",
            )
            logger.exception("视频任务未处理异常 task_id=%s", task_id)

    async def process_next(self) -> bool:
        task = await asyncio.to_thread(
            self._repository.claim_next_video_task,
            self.worker_id,
        )
        if task is None:
            return False
        await self.process_task(task)
        return True

    async def run_forever(self) -> None:
        recovered = await asyncio.to_thread(
            self._repository.recover_stale_video_tasks,
            lease_seconds=self._settings.worker_lease_seconds,
            max_attempts=self._settings.worker_max_attempts,
        )
        cleaned = await asyncio.to_thread(self.cleanup_expired_frames)
        logger.info(
            "视频 Worker 启动 worker_id=%s recovered=%s failed=%s cleaned_frames=%s",
            self.worker_id,
            recovered["recovered"],
            recovered["failed"],
            cleaned,
        )
        while not self._stop.is_set():
            processed = await self.process_next()
            if processed:
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self._settings.worker_poll_interval_seconds,
                )
            except TimeoutError:
                pass
        logger.info("视频 Worker 已停止 worker_id=%s", self.worker_id)


async def _main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=logging.DEBUG if settings.app_debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    storage = StorageService(settings)
    repository = SQLiteRepository(settings.database_path)
    processor = VideoProcessor(settings)
    provider = create_vision_provider(settings)
    image_service = ImageService(settings)
    legacy = AnalysisService(provider, settings)
    image_analysis = ImageAnalysisService(
        provider,
        legacy,
        settings,
        prompt_manager=PromptManager(settings.prompts_path),
    )
    worker = VideoTaskWorker(
        settings,
        storage,
        repository,
        processor,
        image_service,
        image_analysis,
    )
    loop = asyncio.get_running_loop()
    for signal_name in ("SIGTERM", "SIGINT"):
        try:
            import signal

            loop.add_signal_handler(getattr(signal, signal_name), worker.stop)
        except (ImportError, NotImplementedError):
            break
    try:
        await worker.run_forever()
    finally:
        await provider.aclose()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
