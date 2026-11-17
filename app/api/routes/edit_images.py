"""编辑图片上传、分析与人物查询 API。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.api.deps import (
    get_edit_analysis_service,
    get_edit_image_service,
    get_edit_repository,
)
from app.core.exceptions import ResourceNotFoundError
from app.repositories.base import EditRepository
from app.schemas.analysis import ErrorResponse
from app.schemas.edit import (
    EditAnalyzeData,
    EditAnalyzeResponse,
    ImageUploadResponse,
    ImageView,
    PersonListResponse,
    PersonView,
    RegionListResponse,
    RegionView,
)
from app.services.edit_analysis_service import EditAnalysisService
from app.services.edit_image_service import EditImageService

router = APIRouter(prefix="/api/v1/edit", tags=["图像编辑"])


@router.post(
    "/images",
    response_model=ImageUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="上传编辑图片",
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def upload_edit_image(
    file: UploadFile = File(..., description="待编辑的人物图片"),
    image_service: EditImageService = Depends(get_edit_image_service),
) -> ImageUploadResponse:
    try:
        record = await image_service.create_from_upload(file)
    finally:
        await file.close()
    return ImageUploadResponse(
        message="上传成功",
        data=ImageView.from_record(record),
    )


@router.post(
    "/images/{image_id}/analyze",
    response_model=EditAnalyzeResponse,
    summary="识别编辑图片中的人物与可编辑区域",
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def analyze_edit_image(
    image_id: str,
    analysis_service: EditAnalysisService = Depends(get_edit_analysis_service),
) -> EditAnalyzeResponse:
    image, persons, regions = await analysis_service.analyze(image_id)
    return EditAnalyzeResponse(
        message="分析成功",
        data=EditAnalyzeData(
            image=ImageView.from_record(image),
            persons=[PersonView.from_record(person) for person in persons],
            regions=[RegionView.from_record(region) for region in regions],
        ),
    )


@router.get(
    "/images/{image_id}/persons",
    response_model=PersonListResponse,
    summary="获取图片中的人物",
    responses={404: {"model": ErrorResponse}},
)
async def list_image_persons(
    image_id: str,
    image_service: EditImageService = Depends(get_edit_image_service),
    repository: EditRepository = Depends(get_edit_repository),
) -> PersonListResponse:
    await image_service.get_image(image_id)
    persons = await asyncio.to_thread(repository.list_persons, image_id)
    return PersonListResponse(
        data=[PersonView.from_record(person) for person in persons],
    )


@router.get(
    "/persons/{person_id}/regions",
    response_model=RegionListResponse,
    summary="获取人物可编辑区域",
    responses={404: {"model": ErrorResponse}},
)
async def list_person_regions(
    person_id: str,
    repository: EditRepository = Depends(get_edit_repository),
) -> RegionListResponse:
    person = await asyncio.to_thread(repository.get_person, person_id)
    if person is None:
        raise ResourceNotFoundError(detail="人物不存在")
    regions = await asyncio.to_thread(repository.list_regions, person_id)
    return RegionListResponse(
        data=[RegionView.from_record(region) for region in regions],
    )
