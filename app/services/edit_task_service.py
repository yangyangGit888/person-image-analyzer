"""编辑任务（待生成方案）管理。"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

from app.core.exceptions import (
    EditConflictError,
    EditValidationError,
    MaskGenerationError,
    ResourceNotFoundError,
)
from app.repositories.base import EditRepository
from app.schemas.edit import (
    CreateEditTaskRequest,
    EditOperation,
    EditTaskRecord,
    ExistenceStatus,
    MaskStrategy,
    RegionRecord,
    TaskStatus,
    UpdateEditTaskRequest,
)
from app.services.prompt_builder import PromptBuilder


class EditTaskService:
    def __init__(
        self,
        repository: EditRepository,
        prompt_builder: PromptBuilder,
    ) -> None:
        self._repository = repository
        self._prompt_builder = prompt_builder

    async def create(self, request: CreateEditTaskRequest) -> EditTaskRecord:
        image = await asyncio.to_thread(self._repository.get_image, request.image_id)
        if image is None:
            raise ResourceNotFoundError(detail="图片不存在或已过期")
        person = await asyncio.to_thread(
            self._repository.get_person, request.person_id
        )
        if person is None or person.image_id != request.image_id:
            raise ResourceNotFoundError(detail="人物不存在或不属于该图片")
        regions = await asyncio.to_thread(
            self._repository.list_regions, request.person_id
        )
        region = next(
            (
                item
                for item in regions
                if item.region_type == request.region_type
            ),
            None,
        )
        if region is None:
            raise ResourceNotFoundError(detail="该人物没有指定的可编辑区域")
        if region.image_id != request.image_id:
            raise EditValidationError(detail="区域不属于该图片")

        existing = await asyncio.to_thread(
            self._repository.get_task_for_region,
            request.person_id,
            region.region_id,
        )
        if existing is not None:
            raise EditConflictError(
                detail=(
                    "该人物此部位已有编辑任务，请使用 PATCH 修改或先删除旧任务；"
                    f"task_id={existing.task_id}"
                )
            )

        mask_path = self._validate_operation(region, request.operation)
        prompt = self._prompt_builder.build(
            person_ordinal=person.ordinal,
            region_type=region.region_type,
            operation=request.operation,
            style_id=request.style_id,
            color_id=request.color_id,
        )
        now = datetime.now(UTC)
        task = EditTaskRecord(
            task_id=uuid.uuid4().hex,
            image_id=request.image_id,
            person_id=request.person_id,
            region_id=region.region_id,
            region_type=region.region_type,
            operation=request.operation,
            selected_options={
                "style_id": request.style_id,
                "color_id": request.color_id,
            },
            prompt=prompt,
            mask_path=mask_path,
            task_status=TaskStatus.READY,
            created_at=now,
            updated_at=now,
        )
        await asyncio.to_thread(self._repository.save_task, task)
        return task

    async def update(
        self,
        task_id: str,
        request: UpdateEditTaskRequest,
    ) -> EditTaskRecord:
        task = await self.get(task_id)
        if task.task_status is TaskStatus.RUNNING:
            raise EditConflictError(detail="任务正在生成，暂时不能修改")
        region = await asyncio.to_thread(self._repository.get_region, task.region_id)
        if region is None:
            raise ResourceNotFoundError(detail="区域不存在")
        person = await asyncio.to_thread(
            self._repository.get_person, task.person_id
        )
        if person is None:
            raise ResourceNotFoundError(detail="人物不存在")

        operation = request.operation or task.operation
        style_id = request.style_id or task.selected_options.get("style_id", "")
        color_id = request.color_id or task.selected_options.get("color_id", "")
        mask_path = self._validate_operation(region, operation)
        prompt = self._prompt_builder.build(
            person_ordinal=person.ordinal,
            region_type=task.region_type,
            operation=operation,
            style_id=style_id,
            color_id=color_id,
        )

        task.operation = operation
        task.selected_options = {"style_id": style_id, "color_id": color_id}
        task.prompt = prompt
        task.mask_path = mask_path
        task.task_status = TaskStatus.READY
        task.provider_task_id = None
        task.result_path = None
        task.error_code = None
        task.error_detail = None
        task.updated_at = datetime.now(UTC)
        await asyncio.to_thread(self._repository.save_task, task)
        return task

    async def delete(self, task_id: str) -> None:
        task = await self.get(task_id)
        if task.task_status is TaskStatus.RUNNING:
            raise EditConflictError(detail="任务正在生成，不能删除")
        await asyncio.to_thread(self._repository.delete_task, task_id)

    async def get(self, task_id: str) -> EditTaskRecord:
        task = await asyncio.to_thread(self._repository.get_task, task_id)
        if task is None:
            raise ResourceNotFoundError(detail="编辑任务不存在")
        return task

    async def list(
        self,
        *,
        image_id: str | None = None,
        person_id: str | None = None,
    ) -> list[EditTaskRecord]:
        return await asyncio.to_thread(
            self._repository.list_tasks,
            image_id=image_id,
            person_id=person_id,
        )

    @staticmethod
    def _validate_operation(region: RegionRecord, operation: EditOperation) -> str:
        if operation not in {EditOperation.ADD, EditOperation.REPLACE}:
            raise EditValidationError(detail="当前版本仅支持添加和替换操作")
        if operation is EditOperation.ADD:
            if region.existence_status is ExistenceStatus.EXISTING:
                raise EditConflictError(detail="该部位已经存在，请使用替换操作")
        elif operation is EditOperation.REPLACE:
            if region.existence_status is ExistenceStatus.MISSING:
                raise EditConflictError(detail="该部位不存在，请使用添加操作")
            if region.mask_strategy is not MaskStrategy.OBJECT:
                raise MaskGenerationError(
                    detail="该部位没有可用于替换的物体 Mask，请改用添加操作",
                )
        if not region.mask_path:
            raise MaskGenerationError(detail="该区域没有可用 Mask，请重新分析图片")
        return region.mask_path
