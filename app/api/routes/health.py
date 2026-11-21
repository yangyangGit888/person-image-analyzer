"""健康检查路由。

本地 API、SQLite、存储目录、FFmpeg、DeepSeek 配置和 Worker 分别报告状态。
DeepSeek 暂时不可用不会让本地 API 的整体健康状态变成不可用。
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends

from app.ai.base import BaseVisionProvider
from app.api.deps import get_app_settings, get_repository, get_storage_service, get_vision_provider
from app.core.config import Settings
from app.repositories.sqlite_repository import SQLiteRepository
from app.schemas.system import (
    ComponentHealth,
    HealthComponents,
    HealthResponse,
)
from app.services.storage_service import StorageService

router = APIRouter(tags=["系统"])


async def _sqlite_component(repository: SQLiteRepository) -> ComponentHealth:
    started = time.perf_counter()
    ok, detail = await asyncio.to_thread(repository.healthcheck)
    latency = round((time.perf_counter() - started) * 1000, 2)
    return ComponentHealth(
        status="ok" if ok else "unavailable",
        detail=detail,
        latency_ms=latency,
    )


async def _storage_component(storage: StorageService) -> ComponentHealth:
    started = time.perf_counter()
    ok, detail = await asyncio.to_thread(storage.healthcheck)
    latency = round((time.perf_counter() - started) * 1000, 2)
    return ComponentHealth(
        status="ok" if ok else "unavailable",
        detail=detail,
        latency_ms=latency,
    )


def _ffmpeg_component(settings: Settings) -> ComponentHealth:
    path = settings.ffmpeg_path
    if not path:
        return ComponentHealth(
            status="unavailable",
            detail="未找到 ffmpeg；视频分析暂不可用",
        )
    if not os.access(path, os.X_OK):
        return ComponentHealth(
            status="unavailable",
            detail="ffmpeg 文件不可执行",
        )
    return ComponentHealth(status="ok", detail=path)


def _ai_component(
    settings: Settings,
    provider: BaseVisionProvider,
) -> ComponentHealth:
    if not settings.api_key_configured:
        return ComponentHealth(status="unavailable", detail="未配置模型凭证")
    if not provider.supports_vision:
        return ComponentHealth(
            status="degraded",
            detail=f"模型 {provider.model} 不支持视觉输入",
        )
    return ComponentHealth(
        status="ok",
        detail=f"{provider.provider_name}/{provider.model}",
    )


def _worker_component(settings: Settings) -> ComponentHealth:
    pid_file = Path(settings.project_root) / ".run" / "worker.pid"
    if not pid_file.is_file():
        return ComponentHealth(
            status="disabled",
            detail="视频 Worker 未启用（Phase 3）",
        )
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return ComponentHealth(status="degraded", detail="Worker PID 文件无效")
    try:
        os.kill(pid, 0)
    except OSError:
        return ComponentHealth(status="degraded", detail="Worker 进程未运行")
    return ComponentHealth(status="ok", detail=f"worker pid={pid}")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="服务健康检查",
)
async def health_check(
    settings: Settings = Depends(get_app_settings),
    provider: BaseVisionProvider = Depends(get_vision_provider),
    repository: SQLiteRepository = Depends(get_repository),
    storage: StorageService = Depends(get_storage_service),
) -> HealthResponse:
    """返回分组件服务状态，不返回任何密钥。"""

    sqlite_health, storage_health = await asyncio.gather(
        _sqlite_component(repository),
        _storage_component(storage),
    )
    components = HealthComponents(
        api=ComponentHealth(status="ok", detail="FastAPI 服务运行中"),
        sqlite=sqlite_health,
        storage=storage_health,
        ffmpeg=_ffmpeg_component(settings),
        ai=_ai_component(settings, provider),
        worker=_worker_component(settings),
    )
    local_ok = sqlite_health.status == "ok" and storage_health.status == "ok"

    return HealthResponse(
        status="ok" if local_ok else "degraded",
        service=settings.app_name,
        version=settings.app_version,
        ai_configured=settings.api_key_configured,
        timestamp=datetime.now(UTC),
        components=components,
    )
