"""FastAPI 应用入口。

启动方式：
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.ai.base import BaseVisionProvider
from app.ai.factory import create_vision_provider
from app.api.routes import analyze, health, model
from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.schemas.analysis import ErrorDetail, ErrorResponse
from app.services.analysis_service import AnalysisService
from app.services.image_service import ImageService

logger = logging.getLogger(__name__)


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
        app.state.settings = app_settings
        app.state.ai_provider = provider
        app.state.image_service = ImageService(app_settings)
        app.state.analysis_service = AnalysisService(provider, app_settings)

        logger.info(
            "服务启动: provider=%s model=%s base_url=%s api_key_configured=%s",
            provider.provider_name,
            provider.model,
            provider.base_url,
            app_settings.api_key_configured,
        )
        if not app_settings.api_key_configured:
            logger.warning("未配置 AI_API_KEY，图片分析接口将返回配置错误")

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

    app.include_router(health.router)
    app.include_router(analyze.router)
    app.include_router(model.router)

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
        _: Request,
        exc: AppError,
    ) -> JSONResponse:
        log = logger.error if exc.status_code >= 500 else logger.warning
        log("业务异常: code=%s status=%s detail=%s", exc.code, exc.status_code, exc.detail)
        payload = ErrorResponse(
            success=False,
            message=exc.message,
            error=ErrorDetail(code=exc.code, detail=exc.detail),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        errors = exc.errors()
        detail = "请求参数校验失败"
        if errors:
            first = errors[0]
            location = ".".join(
                str(item) for item in first.get("loc", ()) if item != "body"
            )
            message = str(first.get("msg", ""))
            detail = f"请求参数校验失败：{location} {message}".strip()
        payload = ErrorResponse(
            success=False,
            message="请求参数错误",
            error=ErrorDetail(code="VALIDATION_ERROR", detail=detail),
        )
        return JSONResponse(
            status_code=422,
            content=payload.model_dump(mode="json"),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        _: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        payload = ErrorResponse(
            success=False,
            message="请求失败",
            error=ErrorDetail(code="HTTP_ERROR", detail=str(exc.detail)),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.exception("未处理异常: path=%s", request.url.path)
        payload = ErrorResponse(
            success=False,
            message="服务器内部错误",
            error=ErrorDetail(
                code="INTERNAL_ERROR",
                detail="服务器处理请求时发生异常，请稍后重试",
            ),
        )
        return JSONResponse(
            status_code=500,
            content=payload.model_dump(mode="json"),
        )

    return app


app = create_app()
