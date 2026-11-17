"""图像编辑生成任务 API。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from fastapi.responses import FileResponse

from app.api.deps import get_generation_service
from app.schemas.analysis import ErrorResponse
from app.schemas.edit import (
    CreateGenerationRequest,
    GenerationData,
    GenerationResponse,
    GenerationStepView,
    GenerationView,
)
from app.services.generation_service import GenerationService

router = APIRouter(prefix="/api/v1/edit", tags=["图像编辑"])


@router.post(
    "/generations",
    response_model=GenerationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="提交最终生成计划",
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def create_generation(
    request: CreateGenerationRequest,
    service: GenerationService = Depends(get_generation_service),
) -> GenerationResponse:
    generation, steps, created = await service.create_generation(request)
    if created:
        service.start(generation.generation_id)
    return GenerationResponse(
        message="已提交生成" if created else "已存在相同生成任务",
        data=GenerationData(
            generation=GenerationView.from_record(generation),
            steps=[GenerationStepView.from_record(step) for step in steps],
        ),
    )


@router.get(
    "/generations/{generation_id}",
    response_model=GenerationResponse,
    summary="查询生成状态与步骤",
    responses={404: {"model": ErrorResponse}},
)
async def get_generation(
    generation_id: str,
    service: GenerationService = Depends(get_generation_service),
) -> GenerationResponse:
    generation = await service.get_generation(generation_id)
    steps = await service.get_steps(generation_id)
    return GenerationResponse(
        data=GenerationData(
            generation=GenerationView.from_record(generation),
            steps=[GenerationStepView.from_record(step) for step in steps],
        ),
    )


@router.get(
    "/generations/{generation_id}/result",
    summary="下载最终编辑结果",
    responses={
        200: {"content": {"image/png": {}}},
        404: {"model": ErrorResponse},
    },
)
async def download_generation_result(
    generation_id: str,
    service: GenerationService = Depends(get_generation_service),
) -> FileResponse:
    path = await service.get_result_path(generation_id)
    return FileResponse(
        path=path,
        media_type="image/png",
        filename=f"{generation_id}.png",
    )
