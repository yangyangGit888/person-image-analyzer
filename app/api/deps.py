"""FastAPI 依赖注入。

服务实例在应用 lifespan 中创建并挂载到 ``app.state``，便于测试替换和复用
HTTP 连接池。
"""

from __future__ import annotations

from fastapi import Request

from app.ai.base import BaseVisionProvider
from app.ai.prompt_manager import PromptManager
from app.core.config import Settings
from app.repositories.sqlite_repository import SQLiteRepository
from app.services.analysis_service import AnalysisService
from app.services.image_analysis_service import ImageAnalysisService
from app.services.image_service import ImageService
from app.services.storage_service import StorageService
from app.services.video_analysis_service import VideoAnalysisService


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_image_service(request: Request) -> ImageService:
    return request.app.state.image_service


def get_analysis_service(request: Request) -> AnalysisService:
    return request.app.state.analysis_service


def get_image_analysis_service(request: Request) -> ImageAnalysisService:
    return request.app.state.image_analysis_service


def get_vision_provider(request: Request) -> BaseVisionProvider:
    return request.app.state.ai_provider


def get_repository(request: Request) -> SQLiteRepository:
    return request.app.state.repository


def get_storage_service(request: Request) -> StorageService:
    return request.app.state.storage_service


def get_prompt_manager(request: Request) -> PromptManager:
    return request.app.state.prompt_manager


def get_video_analysis_service(request: Request) -> VideoAnalysisService:
    return request.app.state.video_analysis_service
