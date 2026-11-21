"""图片分析 API 路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Request, UploadFile, status

from app.api.deps import get_image_analysis_service, get_image_service
from app.schemas.analysis import AnalyzeResponse, ErrorResponse
from app.services.image_analysis_service import ImageAnalysisService
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
    request: Request,
    file: UploadFile = File(..., description="待分析的图片文件"),
    image_service: ImageService = Depends(get_image_service),
    image_analysis_service: ImageAnalysisService = Depends(get_image_analysis_service),
) -> AnalyzeResponse:
    """上传图片并返回结构化人物属性。"""

    request_id = getattr(request.state, "request_id", None)
    try:
        raw = await image_service.read_upload(file)
    finally:
        await file.close()

    processed = await image_service.process(raw)
    outcome = await image_analysis_service.analyze(
        processed,
        filename=file.filename,
        request_id=request_id,
    )
    return AnalyzeResponse(
        success=True,
        message="分析成功",
        data=outcome.data,
    )
