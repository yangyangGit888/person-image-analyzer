"""FastAPI 应用入口。

启动方式：
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.ai.base import BaseVisionProvider
from app.ai.factory import create_vision_provider
from app.ai.prompt_manager import PromptManager
from app.api.routes import analyze, health, model, video_tasks
from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.processing.video_processor import VideoProcessor
from app.repositories.sqlite_repository import SQLiteRepository
from app.schemas.analysis import ErrorDetail, ErrorResponse
from app.services.analysis_service import AnalysisService
from app.services.image_analysis_service import ImageAnalysisService
from app.services.image_service import ImageService
from app.services.storage_service import StorageService
from app.services.video_analysis_service import VideoAnalysisService

logger = logging.getLogger(__name__)

_REQUEST_ID_MAX_LENGTH = 128


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    if isinstance(value, str) and value:
        return value
    return uuid.uuid4().hex


def _safe_request_id(value: str | None) -> str:
    if not value:
        return uuid.uuid4().hex
    cleaned = value.strip()
    if not cleaned or len(cleaned) > _REQUEST_ID_MAX_LENGTH:
        return uuid.uuid4().hex
    if not all(char.isalnum() or char in "-_.:" for char in cleaned):
        return uuid.uuid4().hex
    return cleaned


def _configure_logging(settings: Settings) -> None:
    level = logging.DEBUG if settings.app_debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("app").setLevel(level)


def create_app(settings: Settings | None = None) -> FastAPI:
    """创建并配置 FastAPI 应用，支持测试时注入自定义 Settings。"""

    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _configure_logging(app_settings)
        provider: BaseVisionProvider = create_vision_provider(app_settings)
        storage = StorageService(app_settings)
        repository = SQLiteRepository(app_settings.database_path)
        prompt_manager = PromptManager(app_settings.prompts_path)

        app.state.settings = app_settings
        app.state.ai_provider = provider
        app.state.image_service = ImageService(app_settings)
        app.state.analysis_service = AnalysisService(provider, app_settings)
        app.state.image_analysis_service = ImageAnalysisService(
            provider,
            app.state.analysis_service,
            app_settings,
            prompt_manager=prompt_manager,
        )
        video_processor = VideoProcessor(app_settings)
        app.state.video_processor = video_processor
        app.state.video_analysis_service = VideoAnalysisService(
            app_settings,
            storage,
            repository,
            video_processor,
        )
        app.state.storage_service = storage
        app.state.repository = repository
        app.state.prompt_manager = prompt_manager
        app.state.started_at = datetime.now(UTC)

        recovered = await asyncio.to_thread(
            repository.recover_stale_video_tasks,
            lease_seconds=app_settings.worker_lease_seconds,
            max_attempts=app_settings.worker_max_attempts,
        )
        if recovered["recovered"] or recovered["failed"]:
            logger.warning(
                "启动恢复视频任务: recovered=%s failed=%s",
                recovered["recovered"],
                recovered["failed"],
            )

        expired = await asyncio.to_thread(repository.cleanup_expired)
        for relative_path in expired["paths"]:
            await asyncio.to_thread(storage.delete_file, relative_path)
        if any(expired[key] for key in ("analyses", "tasks", "masks")):
            logger.info(
                "清理过期数据: analyses=%s tasks=%s masks=%s",
                expired["analyses"],
                expired["tasks"],
                expired["masks"],
            )

        logger.info(
            "服务启动: provider=%s model=%s base_url=%s ai_configured=%s "
            "storage=%s database=%s",
            provider.provider_name,
            provider.model,
            provider.base_url,
            app_settings.api_key_configured,
            app_settings.storage_path,
            app_settings.database_path,
        )
        if not app_settings.api_key_configured:
            logger.warning("未配置模型凭证，图片分析接口将返回配置错误")

        try:
            yield
        finally:
            await provider.aclose()
            logger.info("服务已停止")

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description=(
            "上传人物图片，调用视觉大模型识别穿着属性。"
            "所有三态字段使用 true / false / unknown 字符串。"
        ),
        lifespan=lifespan,
    )
    app.state.settings = app_settings

    if app_settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=app_settings.cors_origin_list,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def limit_upload_request_size(request: Request, call_next):
        """在 multipart 解析前拦截明显超限的请求。"""

        if request.url.path == "/api/v1/analyze":
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    declared_length = int(content_length)
                except ValueError:
                    declared_length = 0
                allowed = app_settings.max_upload_size_bytes + 1024 * 1024
                if declared_length > allowed:
                    payload = ErrorResponse(
                        success=False,
                        message="图片过大",
                        error=ErrorDetail(
                            code="IMAGE_TOO_LARGE",
                            detail=(
                                "图片大小不能超过 "
                                f"{app_settings.max_upload_size_mb}MB"
                            ),
                        ),
                    )
                    return JSONResponse(
                        status_code=413,
                        content=payload.model_dump(mode="json"),
                    )
        return await call_next(request)

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        """为每个请求生成或透传 request_id，并写入响应头。"""

        request_id = _safe_request_id(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "请求完成 request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    app.include_router(health.router)
    app.include_router(analyze.router)
    app.include_router(model.router)
    app.include_router(video_tasks.router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": app_settings.app_name,
            "version": app_settings.app_version,
            "docs": "/docs",
            "health": "/health",
            "analyze": "/api/v1/analyze",
        }

    @app.exception_handler(AppError)
    async def handle_app_error(
        request: Request,
        exc: AppError,
    ) -> JSONResponse:
        request_id = _request_id(request)
        log = logger.error if exc.status_code >= 500 else logger.warning
        log(
            "业务异常: request_id=%s code=%s status=%s detail=%s",
            request_id,
            exc.code,
            exc.status_code,
            exc.detail,
        )
        payload = ErrorResponse(
            success=False,
            message=exc.message,
            error=ErrorDetail(code=exc.code, detail=exc.detail),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(mode="json", exclude_none=True),
            headers={"X-Request-ID": request_id},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request_id = _request_id(request)
        errors = exc.errors()
        detail = "请求参数校验失败"
        details: list[dict[str, Any]] = []
        if errors:
            first = errors[0]
            location = ".".join(
                str(item) for item in first.get("loc", ()) if item != "body"
            )
            message = str(first.get("msg", ""))
            detail = f"请求参数校验失败：{location} {message}".strip()
            for error in errors[:10]:
                details.append(
                    {
                        "loc": [
                            str(item)
                            for item in error.get("loc", ())
                            if item != "body"
                        ],
                        "msg": str(error.get("msg", "")),
                        "type": str(error.get("type", "validation_error")),
                    }
                )
        logger.warning(
            "请求参数校验失败: request_id=%s detail=%s",
            request_id,
            detail,
        )
        payload = ErrorResponse(
            success=False,
            message="请求参数错误",
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                detail=detail,
                details=details or None,
            ),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=422,
            content=payload.model_dump(mode="json", exclude_none=True),
            headers={"X-Request-ID": request_id},
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        request_id = _request_id(request)
        payload = ErrorResponse(
            success=False,
            message="请求失败",
            error=ErrorDetail(code="HTTP_ERROR", detail=str(exc.detail)),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(mode="json", exclude_none=True),
            headers={"X-Request-ID": request_id},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        request_id = _request_id(request)
        logger.exception(
            "未处理异常: request_id=%s path=%s",
            request_id,
            request.url.path,
        )
        payload = ErrorResponse(
            success=False,
            message="服务器内部错误",
            error=ErrorDetail(
                code="INTERNAL_ERROR",
                detail="服务器处理请求时发生异常，请稍后重试",
            ),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=500,
            content=payload.model_dump(mode="json", exclude_none=True),
            headers={"X-Request-ID": request_id},
        )

    return app


app = create_app()
