"""FastAPI 依赖注入。

服务实例在应用 lifespan 中创建并挂载到 ``app.state``，便于测试替换和复用
HTTP 连接池。
"""

from __future__ import annotations

from fastapi import Request

from app.ai.base import BaseVisionProvider
from app.core.config import Settings
from app.repositories.base import EditRepository
from app.services.analysis_service import AnalysisService
from app.services.edit_analysis_service import EditAnalysisService
from app.services.edit_image_service import EditImageService
from app.services.edit_task_service import EditTaskService
from app.services.generation_service import GenerationService
from app.services.image_service import ImageService
from app.services.mask_service import MaskService


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_image_service(request: Request) -> ImageService:
    return request.app.state.image_service


def get_analysis_service(request: Request) -> AnalysisService:
    return request.app.state.analysis_service


def get_vision_provider(request: Request) -> BaseVisionProvider:
    return request.app.state.ai_provider


def get_edit_image_service(request: Request) -> EditImageService:
    return request.app.state.edit_image_service


def get_edit_analysis_service(request: Request) -> EditAnalysisService:
    return request.app.state.edit_analysis_service


def get_edit_task_service(request: Request) -> EditTaskService:
    return request.app.state.edit_task_service


def get_generation_service(request: Request) -> GenerationService:
    return request.app.state.generation_service


def get_edit_repository(request: Request) -> EditRepository:
    return request.app.state.edit_repository


def get_mask_service(request: Request) -> MaskService:
    return request.app.state.mask_service
