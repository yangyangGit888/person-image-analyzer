"""FastAPI 依赖注入。

服务实例在应用 lifespan 中创建并挂载到 ``app.state``，便于测试替换和复用
HTTP 连接池。
"""

from __future__ import annotations

from fastapi import Request

from app.ai.base import BaseVisionProvider
from app.core.config import Settings
from app.services.analysis_service import AnalysisService
from app.services.image_service import ImageService


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_image_service(request: Request) -> ImageService:
    return request.app.state.image_service


def get_analysis_service(request: Request) -> AnalysisService:
    return request.app.state.analysis_service


def get_vision_provider(request: Request) -> BaseVisionProvider:
    return request.app.state.ai_provider
