"""视频上传、任务查询、取消与结果读取。"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.core.config import Settings
from app.core.exceptions import (
    InvalidVideoError,
    TaskCancelledError,
    TaskFailedError,
    TaskNotCompletedError,
    TaskNotFoundError,
    VideoTooLargeError,
)
from app.processing.video_processor import VideoProcessor
from app.repositories.sqlite_repository import SQLiteRepository
from app.schemas.analysis_v2 import AnalysisResultV2
from app.schemas.video import (
    TaskError,
    VideoInfo,
    VideoStage,
    VideoTaskStatus,
    VideoTaskView,
)
from app.services.storage_service import StorageService

_ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
_CHUNK_SIZE = 1024 * 1024


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def task_view(record: dict[str, Any]) -> VideoTaskView:
    try:
        status = VideoTaskStatus(str(record.get("status")))
    except ValueError:
        status = VideoTaskStatus.FAILED
    try:
        stage = VideoStage(str(record.get("stage")))
    except ValueError:
        stage = VideoStage.FAILED
    video_info = None
    if isinstance(record.get("video_info"), dict) and record["video_info"]:
        with contextlib.suppress(Exception):
            video_info = VideoInfo.model_validate(record["video_info"])
    error = None
    if record.get("error_code"):
        error = TaskError(
            code=str(record["error_code"]),
            message="视频任务处理失败",
            detail=record.get("error_detail"),
        )
    created = _parse_dt(record.get("created_at")) or datetime.now(UTC)
    updated = _parse_dt(record.get("updated_at")) or created
    return VideoTaskView(
        task_id=str(record["task_id"]),
        status=status,
        stage=stage,
        progress=max(0, min(100, int(record.get("progress") or 0))),
        video_info=video_info,
        error=error,
        warnings=[str(item) for item in record.get("warnings") or []][:50],
        cancel_requested=bool(record.get("cancel_requested")),
        attempts=int(record.get("attempts") or 0),
        created_at=created,
        updated_at=updated,
        started_at=_parse_dt(record.get("started_at")),
        finished_at=_parse_dt(record.get("finished_at")),
        expires_at=_parse_dt(record.get("expires_at")),
        result_analysis_id=record.get("result_analysis_id"),
        usage={
            str(key): int(value)
            for key, value in (record.get("usage") or {}).items()
            if isinstance(value, int) and not isinstance(value, bool)
        },
    )


class VideoAnalysisService:
    def __init__(
        self,
        settings: Settings,
        storage: StorageService,
        repository: SQLiteRepository,
        processor: VideoProcessor,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._repository = repository
        self._processor = processor

    async def _save_upload(self, upload: UploadFile, relative_path: str) -> int:
        target = self._storage.path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        total = 0
        limit = self._settings.video_max_upload_size_bytes
        declared = getattr(upload, "size", None)
        if isinstance(declared, int) and declared > limit:
            raise VideoTooLargeError(
                detail=f"视频不能超过 {self._settings.video_max_upload_size_mb}MB"
            )
        try:
            with temp.open("wb") as file_obj:
                while True:
                    chunk = await upload.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > limit:
                        raise VideoTooLargeError(
                            detail=(
                                "视频不能超过 "
                                f"{self._settings.video_max_upload_size_mb}MB"
                            )
                        )
                    file_obj.write(chunk)
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temp, target)
            os.chmod(target, 0o600)
        except Exception:
            temp.unlink(missing_ok=True)
            raise
        if total <= 0:
            target.unlink(missing_ok=True)
            raise InvalidVideoError(detail="视频文件为空")
        return total

    async def create_task(self, upload: UploadFile) -> dict[str, Any]:
        task_id = uuid.uuid4().hex
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            suffix = ".video"
        relative = self._storage.relative_path(
            "uploads",
            "videos",
            f"{task_id}{suffix}",
        )
        await self._save_upload(upload, relative)
        path = self._storage.path(relative)
        try:
            info = await self._processor.probe(path)
        except Exception:
            self._storage.delete_file(relative)
            raise

        now = datetime.now(UTC)
        expires_at = now + timedelta(hours=self._settings.video_retention_hours)
        return await asyncio.to_thread(
            self._repository.create_video_task,
            task_id=task_id,
            source_path=relative,
            original_filename=upload.filename,
            video_info=info.model_dump(mode="json"),
            created_at=now,
            expires_at=expires_at,
        )

    async def get_task(self, task_id: str) -> dict[str, Any]:
        task = await asyncio.to_thread(self._repository.get_video_task, task_id)
        if task is None:
            raise TaskNotFoundError()
        return task

    async def cancel(self, task_id: str) -> dict[str, Any]:
        task = await asyncio.to_thread(self._repository.request_cancel, task_id)
        if task is None:
            raise TaskNotFoundError()
        return task

    async def get_result(self, task_id: str) -> AnalysisResultV2:
        task = await self.get_task(task_id)
        status = str(task.get("status"))
        if status == "cancelled":
            raise TaskCancelledError()
        if status == "failed":
            raise TaskFailedError(
                detail=str(task.get("error_detail") or "视频任务处理失败"),
            )
        if status != "completed":
            raise TaskNotCompletedError(
                detail=f"任务当前状态：{status}，进度：{task.get('progress', 0)}%",
            )
        analysis_id = task.get("result_analysis_id")
        if not analysis_id:
            raise TaskFailedError(detail="任务已完成但缺少结果记录")
        payload = await asyncio.to_thread(self._repository.get_analysis, analysis_id)
        if payload is None:
            raise TaskFailedError(detail="分析结果不存在或已过期")
        return AnalysisResultV2.model_validate(payload)

    async def delete_task(self, task_id: str) -> None:
        task = await asyncio.to_thread(self._repository.get_video_task, task_id)
        if task is None:
            raise TaskNotFoundError()
        paths = await asyncio.to_thread(self._repository.delete_video_task, task_id)
        for relative_path in paths:
            await asyncio.to_thread(self._storage.delete_file, relative_path)
        await asyncio.to_thread(self._storage.delete_tree, f"frames/{task_id}")
        if task.get("source_path"):
            await asyncio.to_thread(self._storage.delete_file, task["source_path"])
