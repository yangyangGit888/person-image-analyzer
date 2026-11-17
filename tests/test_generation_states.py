"""万相任务失败、超时与重复提交状态测试。"""

from __future__ import annotations

import time
from typing import Any

from app.ai.image_edit.base import (
    BaseImageEditProvider,
    ImageEditRequest,
    ImageEditTask,
)
from app.services.generation_service import GenerationService
from tests.helpers import make_image_bytes


class FailedProvider(BaseImageEditProvider):
    provider_name = "failed-mock"
    model = "failed-mock"

    async def submit(self, request: ImageEditRequest) -> ImageEditTask:
        return ImageEditTask(task_id="failed-task", status="PENDING")

    async def fetch(self, task_id: str) -> ImageEditTask:
        return ImageEditTask(
            task_id=task_id,
            status="FAILED",
            error_code="MockFailed",
            error_message="mock provider failure",
        )

    async def download(self, url: str) -> bytes:
        raise AssertionError("should not download")


class RunningProvider(BaseImageEditProvider):
    provider_name = "running-mock"
    model = "running-mock"

    async def submit(self, request: ImageEditRequest) -> ImageEditTask:
        return ImageEditTask(task_id="running-task", status="PENDING")

    async def fetch(self, task_id: str) -> ImageEditTask:
        return ImageEditTask(task_id=task_id, status="RUNNING")

    async def download(self, url: str) -> bytes:
        raise AssertionError("should not download")


def _setup_task(client) -> tuple[dict[str, Any], dict[str, Any], str]:
    response = client.post(
        "/api/v1/edit/images",
        files={
            "file": (
                "portrait.png",
                make_image_bytes("PNG", (800, 1000)),
                "image/png",
            )
        },
    )
    image = response.json()["data"]
    analysis = client.post(f"/api/v1/edit/images/{image['image_id']}/analyze")
    data = analysis.json()["data"]
    person = data["persons"][0]
    task = client.post(
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
    assert task.status_code == 201, task.text
    return image, person, task.json()["data"]["task_id"]


def _install_service(client, provider) -> GenerationService:
    state = client.app.state
    settings = state.settings.model_copy(
        update={
            "image_edit_timeout": 0.2,
            "image_edit_poll_initial": 0.05,
            "image_edit_poll_max": 0.05,
            "image_edit_max_retries": 1,
        }
    )
    service = GenerationService(
        settings,
        state.edit_repository,
        state.storage_service,
        provider,
    )
    state.generation_service = service
    return service


def _wait(client, generation_id: str):
    result = None
    for _ in range(40):
        response = client.get(f"/api/v1/edit/generations/{generation_id}")
        result = response.json()["data"]
        if result["generation"]["status"] in {"succeeded", "failed"}:
            return result
        time.sleep(0.05)
    return result


def test_provider_failure_marks_generation_failed(make_edit_client) -> None:
    with make_edit_client() as (client, _):
        image, person, task_id = _setup_task(client)
        _install_service(client, FailedProvider())
        response = client.post(
            "/api/v1/edit/generations",
            json={
                "image_id": image["image_id"],
                "person_id": person["person_id"],
                "task_ids": [task_id],
            },
        )
        generation_id = response.json()["data"]["generation"]["generation_id"]
        result = _wait(client, generation_id)

    assert result["generation"]["status"] == "failed"
    assert result["generation"]["error_code"] == "IMAGE_EDIT_FAILED"
    assert result["steps"][0]["status"] == "failed"


def test_provider_timeout_marks_generation_failed(make_edit_client) -> None:
    with make_edit_client() as (client, _):
        image, person, task_id = _setup_task(client)
        _install_service(client, RunningProvider())
        response = client.post(
            "/api/v1/edit/generations",
            json={
                "image_id": image["image_id"],
                "person_id": person["person_id"],
                "task_ids": [task_id],
            },
        )
        generation_id = response.json()["data"]["generation"]["generation_id"]
        result = _wait(client, generation_id)

    assert result["generation"]["status"] == "failed"
    assert result["generation"]["error_code"] == "IMAGE_EDIT_TIMEOUT"


def test_duplicate_active_plan_returns_existing_generation(make_edit_client) -> None:
    with make_edit_client() as (client, _):
        image, person, task_id = _setup_task(client)
        _install_service(client, RunningProvider())
        payload = {
            "image_id": image["image_id"],
            "person_id": person["person_id"],
            "task_ids": [task_id],
        }
        first = client.post("/api/v1/edit/generations", json=payload)
        second = client.post("/api/v1/edit/generations", json=payload)

    assert first.status_code == 202
    assert second.status_code == 202
    assert (
        first.json()["data"]["generation"]["generation_id"]
        == second.json()["data"]["generation"]["generation_id"]
    )
