"""编辑任务 API。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response, status

from app.api.deps import get_edit_task_service
from app.schemas.analysis import ErrorResponse
from app.schemas.edit import (
    CreateEditTaskRequest,
    EditStatusResponse,
    EditTaskListResponse,
    EditTaskResponse,
    EditTaskView,
    UpdateEditTaskRequest,
)
from app.services.edit_task_service import EditTaskService

router = APIRouter(prefix="/api/v1/edit", tags=["图像编辑"])


@router.post(
    "/tasks",
    response_model=EditTaskResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建待生成编辑任务",
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def create_edit_task(
    request: CreateEditTaskRequest,
    service: EditTaskService = Depends(get_edit_task_service),
) -> EditTaskResponse:
    task = await service.create(request)
    return EditTaskResponse(data=EditTaskView.from_record(task))


@router.get(
    "/tasks",
    response_model=EditTaskListResponse,
    summary="查询编辑方案",
)
async def list_edit_tasks(
    image_id: str | None = Query(default=None),
    person_id: str | None = Query(default=None),
    service: EditTaskService = Depends(get_edit_task_service),
) -> EditTaskListResponse:
    tasks = await service.list(image_id=image_id, person_id=person_id)
    return EditTaskListResponse(
        data=[EditTaskView.from_record(task) for task in tasks],
    )


@router.patch(
    "/tasks/{task_id}",
    response_model=EditTaskResponse,
    summary="修改编辑任务",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def update_edit_task(
    task_id: str,
    request: UpdateEditTaskRequest,
    service: EditTaskService = Depends(get_edit_task_service),
) -> EditTaskResponse:
    task = await service.update(task_id, request)
    return EditTaskResponse(data=EditTaskView.from_record(task))


@router.delete(
    "/tasks/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除编辑任务",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def delete_edit_task(
    task_id: str,
    service: EditTaskService = Depends(get_edit_task_service),
) -> Response:
    await service.delete(task_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/tasks/{task_id}/status",
    response_model=EditStatusResponse,
    summary="查询单个编辑任务状态",
    responses={404: {"model": ErrorResponse}},
)
async def get_edit_task_status(
    task_id: str,
    service: EditTaskService = Depends(get_edit_task_service),
) -> EditStatusResponse:
    task = await service.get(task_id)
    return EditStatusResponse(data=EditTaskView.from_record(task))
