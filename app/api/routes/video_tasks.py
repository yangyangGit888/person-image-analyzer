"""视频分析任务 API。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.api.deps import get_video_analysis_service
from app.schemas.analysis import ErrorResponse
from app.schemas.video import (
    VideoTaskCreateData,
    VideoTaskCreateResponse,
    VideoTaskDetailResponse,
    VideoTaskResultResponse,
)
from app.services.video_analysis_service import VideoAnalysisService, task_view

router = APIRouter(prefix="/api/v1", tags=["视频分析"])


@router.post(
    "/analyze-video",
    response_model=VideoTaskCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="创建单人短视频分析任务",
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_video_task(
    file: UploadFile = File(..., description="单人短视频文件"),
    service: VideoAnalysisService = Depends(get_video_analysis_service),
) -> VideoTaskCreateResponse:
    try:
        record = await service.create_task(file)
    finally:
        await file.close()
    view = task_view(record)
    return VideoTaskCreateResponse(
        success=True,
        message="视频任务已创建",
        data=VideoTaskCreateData(
            task_id=view.task_id,
            status=view.status,
            stage=view.stage,
            created_at=view.created_at,
        ),
    )


@router.get(
    "/video-tasks/{task_id}",
    response_model=VideoTaskDetailResponse,
    summary="查询视频任务状态",
)
async def get_video_task(
    task_id: str,
    service: VideoAnalysisService = Depends(get_video_analysis_service),
) -> VideoTaskDetailResponse:
    record = await service.get_task(task_id)
    return VideoTaskDetailResponse(data=task_view(record))


@router.get(
    "/video-tasks/{task_id}/result",
    response_model=VideoTaskResultResponse,
    summary="获取视频融合结果",
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
async def get_video_task_result(
    task_id: str,
    service: VideoAnalysisService = Depends(get_video_analysis_service),
) -> VideoTaskResultResponse:
    analysis = await service.get_result(task_id)
    return VideoTaskResultResponse(data=analysis)


@router.post(
    "/video-tasks/{task_id}/cancel",
    response_model=VideoTaskDetailResponse,
    summary="取消视频任务",
    responses={404: {"model": ErrorResponse}},
)
async def cancel_video_task(
    task_id: str,
    service: VideoAnalysisService = Depends(get_video_analysis_service),
) -> VideoTaskDetailResponse:
    record = await service.cancel(task_id)
    return VideoTaskDetailResponse(message="取消请求已记录", data=task_view(record))
