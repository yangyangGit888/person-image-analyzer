"""验证万相结果只回填 Mask 区域，避免人脸和背景漂移。"""

from __future__ import annotations

import time
from io import BytesIO

from PIL import Image

from app.ai.image_edit.base import (
    BaseImageEditProvider,
    ImageEditRequest,
    ImageEditTask,
)
from app.services.generation_service import GenerationService
from tests.helpers import make_image_bytes


class RecolorProvider(BaseImageEditProvider):
    provider_name = "recolor"
    model = "recolor"

    def __init__(self) -> None:
        self.width = 0
        self.height = 0

    async def submit(self, request: ImageEditRequest) -> ImageEditTask:
        self.width = request.image_width
        self.height = request.image_height
        return ImageEditTask(task_id="recolor-task", status="PENDING")

    async def fetch(self, task_id: str) -> ImageEditTask:
        return ImageEditTask(
            task_id=task_id,
            status="SUCCEEDED",
            result_urls=["mock://recolor"],
        )

    async def download(self, url: str) -> bytes:
        buffer = BytesIO()
        Image.new("RGB", (self.width, self.height), (255, 0, 0)).save(
            buffer, format="PNG"
        )
        return buffer.getvalue()


def test_generation_result_is_composited_with_mask(make_edit_client) -> None:
    with make_edit_client() as (client, _):
        upload = client.post(
            "/api/v1/edit/images",
            files={
                "file": (
                    "portrait.png",
                    make_image_bytes("PNG", (800, 1000)),
                    "image/png",
                )
            },
        )
        image = upload.json()["data"]
        analyzed = client.post(f"/api/v1/edit/images/{image['image_id']}/analyze")
        data = analyzed.json()["data"]
        person = data["persons"][0]
        task_response = client.post(
            "/api/v1/edit/tasks",
            json={
                "image_id": image["image_id"],
                "person_id": person["person_id"],
                "region_type": "hat",
                "operation": "add",
                "style_id": "baseball_cap",
                "color_id": "black",
            },
        )
        task = task_response.json()["data"]

        state = client.app.state
        settings = state.settings.model_copy(
            update={
                "image_edit_timeout": 5,
                "image_edit_poll_initial": 0,
                "image_edit_poll_max": 0,
            }
        )
        provider = RecolorProvider()
        state.generation_service = GenerationService(
            settings,
            state.edit_repository,
            state.storage_service,
            provider,
        )
        generation = client.post(
            "/api/v1/edit/generations",
            json={
                "image_id": image["image_id"],
                "person_id": person["person_id"],
                "task_ids": [task["task_id"]],
            },
        ).json()["data"]["generation"]
        generation_id = generation["generation_id"]

        result = None
        for _ in range(40):
            result = client.get(
                f"/api/v1/edit/generations/{generation_id}"
            ).json()["data"]
            if result["generation"]["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)

        assert result["generation"]["status"] == "succeeded"
        result_bytes = client.get(
            f"/api/v1/edit/generations/{generation_id}/result"
        ).content

        task_record = state.edit_repository.get_task(task["task_id"])
        mask = state.mask_service.png_bytes_to_mask(
            state.storage_service.path(task_record.mask_path).read_bytes()
        )
        base_bytes = state.storage_service.path(
            state.edit_repository.get_image(image["image_id"]).canonical_path
        ).read_bytes()

    base = Image.open(BytesIO(base_bytes)).convert("RGB")
    output = Image.open(BytesIO(result_bytes)).convert("RGB")
    ys, xs = mask.nonzero()
    inside_x, inside_y = int(xs[len(xs) // 2]), int(ys[len(ys) // 2])
    # 远离 Mask 的背景必须保持原图
    assert output.getpixel((10, 10)) == base.getpixel((10, 10))
    # Mask 内部使用万相返回的红色结果
    assert output.getpixel((inside_x, inside_y)) == (255, 0, 0)
