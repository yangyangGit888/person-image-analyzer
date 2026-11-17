"""编辑白名单选项 API。"""

from __future__ import annotations

from fastapi import APIRouter

from app.catalogs.appearance_options import build_options_data
from app.schemas.catalog import AppearanceOptionsResponse

router = APIRouter(prefix="/api/v1/edit", tags=["图像编辑"])


@router.get(
    "/appearance/options",
    response_model=AppearanceOptionsResponse,
    summary="获取可编辑部位与预设选项",
)
async def get_appearance_options() -> AppearanceOptionsResponse:
    return AppearanceOptionsResponse(data=build_options_data())
