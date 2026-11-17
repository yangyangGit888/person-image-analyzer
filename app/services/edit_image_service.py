"""编辑图片上传、规范化与存储。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from io import BytesIO

from fastapi import UploadFile
from PIL import Image
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.exceptions import EditValidationError, ResourceNotFoundError
from app.repositories.base import EditRepository
from app.schemas.edit import AnalysisStatus, StoredImage
from app.services.image_service import ImageService
from app.services.storage_service import StorageService


class EditImageService:
    def __init__(
        self,
        settings: Settings,
        image_service: ImageService,
        storage: StorageService,
        repository: EditRepository,
    ) -> None:
        self._settings = settings
        self._image_service = image_service
        self._storage = storage
        self._repository = repository

    async def create_from_upload(self, upload: UploadFile) -> StoredImage:
        raw = await self._image_service.read_upload(upload)
        processed = await self._image_service.process(raw)
        data, width, height = await run_in_threadpool(
            self._fit_provider_size,
            processed.data,
            processed.processed_width,
            processed.processed_height,
        )

        image_id = self._storage.new_id()
        relative_path = self._storage.relative_path(
            "images", image_id, "source.jpg"
        )
        await self._storage.save_bytes(relative_path, data)

        now = datetime.now(UTC)
        record = StoredImage(
            image_id=image_id,
            original_width=processed.width,
            original_height=processed.height,
            width=width,
            height=height,
            format=processed.source_format,
            file_size=processed.original_size,
            source_path=relative_path,
            canonical_path=relative_path,
            person_count=0,
            analysis_status=AnalysisStatus.PENDING,
            created_at=now,
            expires_at=now + timedelta(hours=self._settings.storage_ttl_hours),
        )
        await asyncio.to_thread(self._repository.save_image, record)
        return record

    def _fit_provider_size(
        self,
        data: bytes,
        width: int,
        height: int,
    ) -> tuple[bytes, int, int]:
        """保证送给万相和分割模型的图片满足 512～2048 的尺寸范围。"""

        min_dimension = self._settings.edit_min_provider_dimension
        max_dimension = self._settings.edit_max_provider_dimension
        if (
            width >= min_dimension
            and height >= min_dimension
            and max(width, height) <= max_dimension
        ):
            return data, width, height

        with Image.open(BytesIO(data)) as image:
            rgb = image.convert("RGB")
            if min(width, height) < min_dimension:
                scale = min_dimension / min(width, height)
                new_size = (round(width * scale), round(height * scale))
            else:
                scale = max_dimension / max(width, height)
                new_size = (round(width * scale), round(height * scale))
            if max(new_size) > max_dimension:
                raise EditValidationError(
                    detail=(
                        "图片宽高比过大，无法同时满足万相最小边 512 和最大边 "
                        f"{max_dimension} 的限制，请先裁剪图片"
                    )
                )
            resized = rgb.resize(new_size, Image.Resampling.LANCZOS)
            output = BytesIO()
            resized.save(output, format="JPEG", quality=92, optimize=True, subsampling=0)
        return output.getvalue(), new_size[0], new_size[1]

    async def get_image(self, image_id: str) -> StoredImage:
        image = await asyncio.to_thread(self._repository.get_image, image_id)
        if image is None:
            raise ResourceNotFoundError(detail="图片不存在或已过期")
        return image

    async def read_image_bytes(self, image: StoredImage) -> bytes:
        return await self._storage.read_bytes(image.canonical_path)

    async def mark_analyzed(self, image: StoredImage, person_count: int) -> StoredImage:
        image.person_count = person_count
        image.analysis_status = AnalysisStatus.COMPLETED
        await asyncio.to_thread(self._repository.update_image, image)
        return image

    async def cleanup_expired(self) -> int:
        """清理过期图片、人物 Mask、区域 Mask 与生成结果。"""

        now = datetime.now(UTC)
        image_ids = await asyncio.to_thread(
            self._repository.list_expired_image_ids, now
        )
        for image_id in image_ids:
            generation_ids = await asyncio.to_thread(
                self._repository.list_generation_ids, image_id
            )
            await self._storage.delete_tree(
                self._storage.relative_path("images", image_id)
            )
            for generation_id in generation_ids:
                await self._storage.delete_tree(
                    self._storage.relative_path("generations", generation_id)
                )
            await asyncio.to_thread(self._repository.delete_image_data, image_id)
        return len(image_ids)
