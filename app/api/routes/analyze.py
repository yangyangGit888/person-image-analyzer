"""图片分析 API 路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.ai.base import ImageInput
from app.api.deps import get_analysis_service, get_image_service
from app.schemas.analysis import AnalyzeResponse, ErrorResponse
from app.services.analysis_service import AnalysisService
from app.services.image_service import ImageService

router = APIRouter(prefix="/api/v1", tags=["图片分析"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="上传图片并识别人物穿着属性",
    description=(
        "接收 multipart/form-data 图片文件，校验真实图片格式后调用视觉大模型。"
        "图片仅在内存中处理，默认不保存到服务器。"
    ),
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        status.HTTP_502_BAD_GATEWAY: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
        status.HTTP_504_GATEWAY_TIMEOUT: {"model": ErrorResponse},
    },
)
async def analyze_image(
    file: UploadFile = File(..., description="待分析的图片文件"),
    image_service: ImageService = Depends(get_image_service),
    analysis_service: AnalysisService = Depends(get_analysis_service),
) -> AnalyzeResponse:
    """上传图片并返回结构化人物属性。"""

    try:
        raw = await image_service.read_upload(file)
    finally:
        await file.close()

    processed = await image_service.process(raw)
    image = ImageInput(
        data=processed.data,
        mime_type=processed.mime_type,
        width=processed.width,
        height=processed.height,
        filename=file.filename,
    )

    data = await analysis_service.analyze(image)
    return AnalyzeResponse(
        success=True,
        message="分析成功",
        data=data,
    )
