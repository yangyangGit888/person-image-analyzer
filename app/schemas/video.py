"""视频任务相关数据结构。"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from app.schemas.analysis_v2 import AnalysisResultV2, SchemaBaseV2


class VideoTaskStatus(StrEnum):
    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    SELECTING_FRAMES = "selecting_frames"
    ANALYZING = "analyzing"
    MERGING = "merging"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VideoStage(StrEnum):
    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    SELECTING_FRAMES = "selecting_frames"
    ANALYZING = "analyzing"
    MERGING = "merging"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VideoInfo(SchemaBaseV2):
    duration_seconds: float = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float | None = Field(default=None, gt=0)
    codec: str | None = None
    format_name: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    has_audio: bool | None = None


class TaskError(SchemaBaseV2):
    code: str
    message: str
    detail: str | None = None


class VideoTaskView(SchemaBaseV2):
    task_id: str
    status: VideoTaskStatus
    stage: VideoStage
    progress: int = Field(default=0, ge=0, le=100)
    video_info: VideoInfo | None = None
    error: TaskError | None = None
    warnings: list[str] = Field(default_factory=list)
    cancel_requested: bool = False
    attempts: int = 0
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    expires_at: datetime | None = None
    result_analysis_id: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)


class VideoTaskCreateData(SchemaBaseV2):
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    status: VideoTaskStatus = VideoTaskStatus.QUEUED
    stage: VideoStage = VideoStage.QUEUED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VideoTaskCreateResponse(SchemaBaseV2):
    success: Literal[True] = True
    message: str = "视频任务已创建"
    data: VideoTaskCreateData


class VideoTaskListData(SchemaBaseV2):
    items: list[VideoTaskView] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class VideoTaskListResponse(SchemaBaseV2):
    success: Literal[True] = True
    message: str = "ok"
    data: VideoTaskListData


class VideoTaskDetailResponse(SchemaBaseV2):
    success: Literal[True] = True
    message: str = "ok"
    data: VideoTaskView


class VideoTaskResultResponse(SchemaBaseV2):
    success: Literal[True] = True
    message: str = "ok"
    data: AnalysisResultV2


class VideoTaskRawRecord(SchemaBaseV2):
    """内部任务记录，允许保存原始 JSON 字段。"""

    task_id: str
    status: str
    stage: str
    progress: int
    source_path: str | None = None
    original_filename: str | None = None
    video_info: dict[str, Any] | None = None
    error_code: str | None = None
    error_detail: str | None = None
    warnings: list[str] = Field(default_factory=list)
    attempts: int = 0
    worker_id: str | None = None
    cancel_requested: bool = False
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    heartbeat_at: str | None = None
    expires_at: str | None = None
    result_analysis_id: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
