"""健康检查路由。"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from app.api.deps import get_app_settings
from app.core.config import Settings
from app.schemas.system import HealthResponse

router = APIRouter(tags=["系统"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="服务健康检查",
)
async def health_check(
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    """返回服务运行状态与 AI 配置是否完整（不返回任何密钥）。"""

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        ai_configured=settings.api_key_configured,
        timestamp=datetime.now(UTC),
    )
