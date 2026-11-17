"""测试与本地联调使用的图像编辑 Provider。"""

from __future__ import annotations

import uuid
from io import BytesIO

from PIL import Image

from app.ai.image_edit.base import (
    BaseImageEditProvider,
    ImageEditRequest,
    ImageEditTask,
)


class MockImageEditProvider(BaseImageEditProvider):
    provider_name = "mock"
    model = "mock-image-edit"

    def __init__(self, *, fail_submit: bool = False, fail_fetch: bool = False) -> None:
        self.fail_submit = fail_submit
        self.fail_fetch = fail_fetch
        self.requests: dict[str, ImageEditRequest] = {}
        self.submit_count = 0
        self.fetch_count = 0

    async def submit(self, request: ImageEditRequest) -> ImageEditTask:
        self.submit_count += 1
        if self.fail_submit:
            from app.core.exceptions import ImageEditFailedError

            raise ImageEditFailedError(detail="Mock 万相提交失败")
        if (request.image_width, request.image_height) != (
            request.mask_width,
            request.mask_height,
        ):
            from app.core.exceptions import EditValidationError

            raise EditValidationError(detail="Mask 尺寸必须与原图完全一致")
        task_id = f"mock-{uuid.uuid4().hex}"
        self.requests[task_id] = request
        return ImageEditTask(task_id=task_id, status="PENDING")

    async def fetch(self, task_id: str) -> ImageEditTask:
        self.fetch_count += 1
        if self.fail_fetch:
            from app.core.exceptions import ImageEditFailedError

            raise ImageEditFailedError(detail="Mock 万相查询失败")
        if task_id not in self.requests:
            return ImageEditTask(
                task_id=task_id,
                status="FAILED",
                error_code="MOCK_TASK_NOT_FOUND",
                error_message="mock task not found",
            )
        return ImageEditTask(
            task_id=task_id,
            status="SUCCEEDED",
            result_urls=[f"mock://result/{task_id}"],
        )

    async def download(self, url: str) -> bytes:
        task_id = url.rsplit("/", 1)[-1]
        request = self.requests.get(task_id)
        if request is None:
            from app.core.exceptions import ImageEditInvalidOutputError

            raise ImageEditInvalidOutputError(detail="Mock 结果不存在")
        with Image.open(BytesIO(request.image_data)) as image:
            output = BytesIO()
            image.convert("RGB").save(output, format="PNG")
        return output.getvalue()
