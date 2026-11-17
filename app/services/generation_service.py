"""多区域顺序生成编排。

第一版采用“每个区域一次万相调用，上一步输出作为下一步输入”的策略，
保留每个区域独立的 Mask 与 Prompt 语义。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageFilter

from app.ai.image_edit.base import BaseImageEditProvider, ImageEditRequest
from app.catalogs.region_definitions import get_generation_order
from app.core.config import Settings
from app.core.exceptions import (
    AppError,
    EditConflictError,
    EditValidationError,
    ImageEditContentRejectedError,
    ImageEditFailedError,
    ImageEditInvalidOutputError,
    ImageEditTimeoutError,
    MaskGenerationError,
    ResourceNotFoundError,
)
from app.repositories.base import EditRepository
from app.schemas.edit import (
    CreateGenerationRequest,
    EditTaskRecord,
    GenerationRecord,
    GenerationStatus,
    GenerationStepRecord,
    StepStatus,
    TaskStatus,
)
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)


class GenerationService:
    def __init__(
        self,
        settings: Settings,
        repository: EditRepository,
        storage: StorageService,
        provider: BaseImageEditProvider,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._storage = storage
        self._provider = provider
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def create_generation(
        self,
        request: CreateGenerationRequest,
    ) -> tuple[GenerationRecord, list[GenerationStepRecord], bool]:
        image = await asyncio.to_thread(self._repository.get_image, request.image_id)
        if image is None:
            raise ResourceNotFoundError(detail="图片不存在或已过期")
        person = await asyncio.to_thread(
            self._repository.get_person, request.person_id
        )
        if person is None or person.image_id != request.image_id:
            raise ResourceNotFoundError(detail="人物不存在或不属于该图片")

        if request.idempotency_key:
            existing = await asyncio.to_thread(
                self._repository.find_generation_by_idempotency,
                request.idempotency_key,
            )
            if existing is not None:
                steps = await asyncio.to_thread(
                    self._repository.list_steps, existing.generation_id
                )
                return existing, steps, False

        tasks = await self._load_tasks(request)
        if len(tasks) > self._settings.edit_max_regions_per_generation:
            raise EditValidationError(
                detail=(
                    "单次生成最多支持 "
                    f"{self._settings.edit_max_regions_per_generation} 个部位"
                )
            )
        ordered = sorted(
            tasks,
            key=lambda task: (
                get_generation_order(task.region_type),
                task.created_at,
            ),
        )
        plan_hash = self._plan_hash(request.image_id, request.person_id, ordered)

        active_same = await asyncio.to_thread(
            self._repository.find_active_generation,
            person_id=request.person_id,
            plan_hash=plan_hash,
        )
        if active_same is not None:
            steps = await asyncio.to_thread(
                self._repository.list_steps, active_same.generation_id
            )
            return active_same, steps, False

        active_any = await asyncio.to_thread(
            self._repository.list_active_generations, request.person_id
        )
        if active_any:
            raise EditConflictError(
                detail="该人物已有生成任务正在执行，请等待完成或稍后重试",
            )

        self._ensure_ready(tasks)

        now = datetime.now(UTC)
        generation = GenerationRecord(
            generation_id=uuid.uuid4().hex,
            image_id=request.image_id,
            person_id=request.person_id,
            task_ids=[task.task_id for task in ordered],
            plan_hash=plan_hash,
            status=GenerationStatus.PENDING,
            current_step=0,
            idempotency_key=request.idempotency_key,
            created_at=now,
            updated_at=now,
        )
        steps = [
            GenerationStepRecord(
                generation_id=generation.generation_id,
                sequence=index,
                task_id=task.task_id,
                region_type=task.region_type,
                status=StepStatus.PENDING,
            )
            for index, task in enumerate(ordered, start=1)
        ]
        await asyncio.to_thread(self._repository.save_generation, generation)
        for step in steps:
            await asyncio.to_thread(self._repository.save_step, step)
        return generation, steps, True

    async def _load_tasks(
        self,
        request: CreateGenerationRequest,
    ) -> list[EditTaskRecord]:
        if request.task_ids:
            tasks: list[EditTaskRecord] = []
            for task_id in dict.fromkeys(request.task_ids):
                task = await asyncio.to_thread(self._repository.get_task, task_id)
                if task is None:
                    raise ResourceNotFoundError(detail=f"编辑任务不存在: {task_id}")
                if task.image_id != request.image_id or task.person_id != request.person_id:
                    raise EditValidationError(detail="编辑任务不属于指定图片或人物")
                tasks.append(task)
            return tasks

        tasks = await asyncio.to_thread(
            self._repository.list_tasks,
            image_id=request.image_id,
            person_id=request.person_id,
        )
        ready = [task for task in tasks if task.task_status is TaskStatus.READY]
        if not ready:
            raise EditValidationError(detail="没有可生成的编辑任务")
        return ready

    @staticmethod
    def _ensure_ready(tasks: list[EditTaskRecord]) -> None:
        for task in tasks:
            if task.task_status is not TaskStatus.READY:
                raise EditConflictError(
                    detail=(
                        f"编辑任务当前不可生成: {task.task_id} "
                        f"({task.task_status.value})"
                    )
                )

    def start(self, generation_id: str) -> None:
        if generation_id in self._tasks:
            return
        task = asyncio.create_task(self.run_generation(generation_id))
        self._tasks[generation_id] = task

        def _done(completed: asyncio.Task[None]) -> None:
            self._tasks.pop(generation_id, None)
            try:
                completed.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("生成后台任务异常: generation_id=%s", generation_id)

        task.add_done_callback(_done)

    async def run_generation(self, generation_id: str) -> GenerationRecord:
        generation = await asyncio.to_thread(
            self._repository.get_generation, generation_id
        )
        if generation is None:
            raise ResourceNotFoundError(detail="生成任务不存在")
        if generation.status is not GenerationStatus.PENDING:
            return generation

        generation.status = GenerationStatus.RUNNING
        generation.updated_at = datetime.now(UTC)
        await asyncio.to_thread(self._repository.save_generation, generation)

        steps = await asyncio.to_thread(self._repository.list_steps, generation_id)
        image = await asyncio.to_thread(
            self._repository.get_image, generation.image_id
        )
        if image is None:
            return await self._fail_generation(
                generation,
                None,
                ResourceNotFoundError(detail="图片不存在或已过期"),
            )

        current_data = await self._storage.read_bytes(image.canonical_path)
        current_path = image.canonical_path
        width, height = image.width, image.height

        for step in steps:
            if step.status is not StepStatus.PENDING:
                continue
            task = await asyncio.to_thread(self._repository.get_task, step.task_id)
            if task is None:
                return await self._fail_generation(
                    generation,
                    step,
                    ResourceNotFoundError(detail="编辑任务不存在"),
                )
            if not task.mask_path:
                return await self._fail_generation(
                    generation,
                    step,
                    MaskGenerationError(detail="编辑任务缺少 Mask"),
                )

            step.status = StepStatus.RUNNING
            step.input_path = current_path
            task.task_status = TaskStatus.RUNNING
            generation.current_step = step.sequence
            generation.updated_at = datetime.now(UTC)
            await asyncio.to_thread(self._repository.update_step, step)
            await asyncio.to_thread(self._repository.save_task, task)
            await asyncio.to_thread(self._repository.save_generation, generation)

            try:
                mask_data = await self._storage.read_bytes(task.mask_path)
                self._validate_mask_size(mask_data, width, height)
                request = ImageEditRequest(
                    image_data=current_data,
                    image_mime="image/jpeg",
                    image_width=width,
                    image_height=height,
                    mask_data=mask_data,
                    mask_mime="image/png",
                    mask_width=width,
                    mask_height=height,
                    prompt=task.prompt,
                    seed=self._settings.image_edit_seed,
                )
                submitted = await self._provider.submit(request)
                task.provider_task_id = submitted.task_id
                step.provider_task_id = submitted.task_id
                await asyncio.to_thread(self._repository.save_task, task)
                await asyncio.to_thread(self._repository.update_step, step)

                completed = await self._wait_for_task(submitted.task_id)
                if not completed.result_urls:
                    raise ImageEditInvalidOutputError(
                        detail="万相任务成功但未返回图片地址",
                    )
                result_bytes = await self._provider.download(completed.result_urls[0])
                normalized = await asyncio.to_thread(
                    self._normalize_output,
                    result_bytes,
                    width,
                    height,
                )
                normalized = await asyncio.to_thread(
                    self._composite_with_mask,
                    current_data,
                    normalized,
                    mask_data,
                    width,
                    height,
                )
                output_path = self._storage.relative_path(
                    "generations",
                    generation.generation_id,
                    f"step_{step.sequence:02d}.png",
                )
                await self._storage.save_bytes(output_path, normalized)

                step.status = StepStatus.SUCCEEDED
                step.output_path = output_path
                task.task_status = TaskStatus.SUCCEEDED
                task.result_path = output_path
                task.error_code = None
                task.error_detail = None
                task.updated_at = datetime.now(UTC)
                generation.updated_at = task.updated_at
                await asyncio.to_thread(self._repository.update_step, step)
                await asyncio.to_thread(self._repository.save_task, task)
                await asyncio.to_thread(self._repository.save_generation, generation)

                current_data = normalized
                current_path = output_path
            except AppError as exc:
                return await self._fail_generation(generation, step, exc)
            except Exception:
                logger.exception("生成步骤异常: generation_id=%s", generation_id)
                return await self._fail_generation(
                    generation,
                    step,
                    ImageEditFailedError(detail="生成过程中发生未预期错误"),
                )

        result_path = self._storage.relative_path(
            "generations", generation.generation_id, "result.png"
        )
        await self._storage.save_bytes(result_path, current_data)
        generation.status = GenerationStatus.SUCCEEDED
        generation.result_path = result_path
        generation.error_code = None
        generation.error_detail = None
        generation.updated_at = datetime.now(UTC)
        await asyncio.to_thread(self._repository.save_generation, generation)
        logger.info(
            "生成完成: generation_id=%s tasks=%s",
            generation.generation_id,
            len(generation.task_ids),
        )
        return generation

    async def _wait_for_task(self, provider_task_id: str):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._settings.image_edit_timeout
        poll = self._settings.image_edit_poll_initial
        while True:
            task = await self._provider.fetch(provider_task_id)
            status = task.status.upper()
            if status == "SUCCEEDED":
                return task
            if status in {"FAILED", "CANCELED", "CANCELLED", "UNKNOWN"}:
                if task.error_code and "inspection" in task.error_code.lower():
                    raise ImageEditContentRejectedError()
                raise ImageEditFailedError(
                    detail=(
                        "万相图像编辑任务失败"
                        + (f"（{task.error_code}）" if task.error_code else "")
                    )
                )
            if loop.time() >= deadline:
                raise ImageEditTimeoutError()
            await asyncio.sleep(max(0.0, poll))
            poll = min(
                self._settings.image_edit_poll_max,
                max(poll * 1.5, self._settings.image_edit_poll_initial),
            )

    async def _fail_generation(
        self,
        generation: GenerationRecord,
        step: GenerationStepRecord | None,
        error: AppError,
    ) -> GenerationRecord:
        now = datetime.now(UTC)
        if step is not None:
            step.status = StepStatus.FAILED
            step.error_code = error.code
            step.error_detail = error.detail
            await asyncio.to_thread(self._repository.update_step, step)
            task = await asyncio.to_thread(
                self._repository.get_task, step.task_id
            )
            if task is not None:
                task.task_status = TaskStatus.FAILED
                task.error_code = error.code
                task.error_detail = error.detail
                task.updated_at = now
                await asyncio.to_thread(self._repository.save_task, task)
        generation.status = GenerationStatus.FAILED
        generation.error_code = error.code
        generation.error_detail = error.detail
        generation.updated_at = now
        await asyncio.to_thread(self._repository.save_generation, generation)
        logger.warning(
            "生成失败: generation_id=%s code=%s step=%s",
            generation.generation_id,
            error.code,
            step.sequence if step else None,
        )
        return generation

    def _validate_mask_size(self, mask_data: bytes, width: int, height: int) -> None:
        try:
            with Image.open(BytesIO(mask_data)) as mask:
                if mask.size != (width, height):
                    raise EditValidationError(
                        detail="Mask 尺寸与当前编辑图片不一致",
                    )
                if mask.mode not in {"L", "1"}:
                    raise EditValidationError(
                        detail="Mask 必须是单通道黑白 PNG",
                    )
        except EditValidationError:
            raise
        except Exception as exc:
            raise MaskGenerationError(detail="Mask 文件无法解析") from exc

    @staticmethod
    def _composite_with_mask(
        base_data: bytes,
        result_data: bytes,
        mask_data: bytes,
        width: int,
        height: int,
    ) -> bytes:
        """只保留 Mask 区域的万相修改，防止人脸和背景漂移。"""

        try:
            with Image.open(BytesIO(base_data)) as base_image:
                base = base_image.convert("RGB")
            with Image.open(BytesIO(result_data)) as result_image:
                result = result_image.convert("RGB")
            with Image.open(BytesIO(mask_data)) as mask_image:
                mask = mask_image.convert("L")
        except Exception as exc:
            raise ImageEditInvalidOutputError(
                detail="无法合成万相编辑结果",
            ) from exc
        if base.size != (width, height) or result.size != (width, height):
            raise ImageEditInvalidOutputError(detail="合成图片尺寸不一致")
        if mask.size != (width, height):
            raise EditValidationError(detail="合成 Mask 尺寸不一致")
        soft_mask = mask.filter(ImageFilter.GaussianBlur(radius=1.2))
        composed = Image.composite(result, base, soft_mask)
        output = BytesIO()
        composed.save(output, format="PNG")
        return output.getvalue()

    @staticmethod
    def _normalize_output(
        data: bytes,
        width: int,
        height: int,
    ) -> bytes:
        try:
            with Image.open(BytesIO(data)) as image:
                rgb = image.convert("RGB")
        except Exception as exc:
            raise ImageEditInvalidOutputError(detail="万相返回的图片无法解析") from exc
        if rgb.size == (width, height):
            output = BytesIO()
            rgb.save(output, format="PNG")
            return output.getvalue()
        source_ratio = rgb.width / float(rgb.height)
        target_ratio = width / float(height)
        if abs(source_ratio / target_ratio - 1.0) > 0.02:
            raise ImageEditInvalidOutputError(
                detail="万相返回图片的宽高比与原图不一致",
            )
        resized = rgb.resize((width, height), Image.Resampling.LANCZOS)
        output = BytesIO()
        resized.save(output, format="PNG")
        return output.getvalue()

    @staticmethod
    def _plan_hash(
        image_id: str,
        person_id: str,
        tasks: list[EditTaskRecord],
    ) -> str:
        payload = {
            "image_id": image_id,
            "person_id": person_id,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "region_type": task.region_type.value,
                    "operation": task.operation.value,
                    "options": task.selected_options,
                    "prompt": task.prompt,
                    "mask_path": task.mask_path,
                }
                for task in tasks
            ],
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    async def get_generation(self, generation_id: str) -> GenerationRecord:
        generation = await asyncio.to_thread(
            self._repository.get_generation, generation_id
        )
        if generation is None:
            raise ResourceNotFoundError(detail="生成任务不存在")
        return generation

    async def get_steps(self, generation_id: str) -> list[GenerationStepRecord]:
        return await asyncio.to_thread(
            self._repository.list_steps, generation_id
        )

    async def get_result_path(self, generation_id: str) -> Path:
        generation = await self.get_generation(generation_id)
        if generation.status is not GenerationStatus.SUCCEEDED or not generation.result_path:
            raise ResourceNotFoundError(detail="生成结果尚未就绪")
        return self._storage.path(generation.result_path)

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        await asyncio.to_thread(
            self._repository.mark_running_generations_interrupted
        )
