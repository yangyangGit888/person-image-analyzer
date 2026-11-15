"""模型配置检查路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.ai.base import BaseVisionProvider
from app.api.deps import get_app_settings, get_vision_provider
from app.core.config import Settings
from app.schemas.system import ModelInfo, ModelInfoResponse

router = APIRouter(prefix="/api/v1", tags=["模型配置"])


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="查看当前视觉模型配置",
    description="返回服务商、模型名称和 API 地址，不返回 API Key。",
)
async def model_info(
    settings: Settings = Depends(get_app_settings),
    provider: BaseVisionProvider = Depends(get_vision_provider),
) -> ModelInfoResponse:
    """返回当前模型配置，敏感信息一律隐藏。"""

    return ModelInfoResponse(
        success=True,
        message="ok",
        data=ModelInfo(
            provider=provider.provider_name,
            model=provider.model,
            base_url=provider.base_url,
            supports_vision=provider.supports_vision,
            api_key_configured=settings.api_key_configured,
            timeout_seconds=settings.ai_timeout,
            max_retries=settings.ai_max_retries,
        ),
    )
