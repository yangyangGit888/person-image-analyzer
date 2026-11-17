"""万相 Provider 的 MockTransport 测试，不消耗真实 API 费用。"""

from __future__ import annotations

import asyncio
import base64
import json
from io import BytesIO
from typing import Any

import httpx
import pytest
from PIL import Image

from app.ai.image_edit.base import ImageEditRequest
from app.ai.image_edit.wanx_provider import WanxImageEditProvider
from app.core.config import Settings
from app.core.exceptions import (
    EditValidationError,
    ImageEditAuthError,
    ImageEditContentRejectedError,
)


def _settings(**overrides: Any) -> Settings:
    config: dict[str, Any] = {
        "dashscope_api_key": "unit-test-dashscope-key",
        "dashscope_base_url": "https://dashscope.aliyuncs.com/api/v1",
        "image_edit_model": "wanx2.1-imageedit",
        "image_edit_max_retries": 0,
        "image_edit_timeout": 5,
    }
    config.update(overrides)
    return Settings(_env_file=None, **config)


def _image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), (100, 120, 140)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _mask_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("L", (32, 32), 255).save(buffer, format="PNG")
    return buffer.getvalue()


def _request() -> ImageEditRequest:
    return ImageEditRequest(
        image_data=_image_bytes(),
        image_mime="image/jpeg",
        image_width=32,
        image_height=32,
        mask_data=_mask_bytes(),
        mask_mime="image/png",
        mask_width=32,
        mask_height=32,
        prompt="给人物添加一顶黑色棒球帽",
    )


def test_submit_payload_and_headers() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"output": {"task_id": "task-1", "task_status": "PENDING"}},
        )

    provider = WanxImageEditProvider(
        _settings(),
        transport=httpx.MockTransport(handler),
    )
    result = asyncio.run(provider.submit(_request()))
    asyncio.run(provider.aclose())

    payload = captured["payload"]
    assert captured["url"].endswith("/services/aigc/image2image/image-synthesis")
    assert captured["headers"]["x-dashscope-async"] == "enable"
    assert captured["headers"]["authorization"] == "Bearer unit-test-dashscope-key"
    assert payload["model"] == "wanx2.1-imageedit"
    assert payload["input"]["function"] == "description_edit_with_mask"
    assert payload["input"]["base_image_url"].startswith("data:image/jpeg;base64,")
    assert payload["input"]["mask_image_url"].startswith("data:image/png;base64,")
    assert payload["parameters"]["n"] == 1
    assert "size" not in payload["parameters"]
    assert result.task_id == "task-1"
    assert result.status == "PENDING"


def test_fetch_and_download() -> None:
    image = _image_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and "/tasks/" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "output": {
                        "task_id": "task-1",
                        "task_status": "SUCCEEDED",
                        "results": [
                            {"url": "https://example.com/result.png"},
                        ],
                    }
                },
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                headers={"content-type": "image/png"},
                content=image,
            )
        raise AssertionError("unexpected request")

    provider = WanxImageEditProvider(
        _settings(),
        transport=httpx.MockTransport(handler),
    )

    async def run() -> tuple[Any, bytes]:
        try:
            status = await provider.fetch("task-1")
            data = await provider.download(status.result_urls[0])
            return status, data
        finally:
            await provider.aclose()

    status, data = asyncio.run(run())
    assert status.status == "SUCCEEDED"
    assert status.result_urls == ["https://example.com/result.png"]
    assert data == image


def test_auth_error_does_not_leak_key() -> None:
    provider = WanxImageEditProvider(
        _settings(),
        transport=httpx.MockTransport(
            lambda _: httpx.Response(401, text="invalid api key")
        ),
    )
    with pytest.raises(ImageEditAuthError) as exc_info:
        asyncio.run(provider.submit(_request()))
    asyncio.run(provider.aclose())
    assert "unit-test-dashscope-key" not in str(exc_info.value)


def test_content_rejection_mapping() -> None:
    provider = WanxImageEditProvider(
        _settings(),
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                400,
                text='{"code":"DataInspectionFailed","message":"content risk"}',
            )
        ),
    )
    with pytest.raises(ImageEditContentRejectedError):
        asyncio.run(provider.submit(_request()))
    asyncio.run(provider.aclose())


def test_mask_size_must_match_image() -> None:
    provider = WanxImageEditProvider(
        _settings(),
        transport=httpx.MockTransport(lambda _: httpx.Response(500)),
    )
    request = ImageEditRequest(
        image_data=_image_bytes(),
        image_mime="image/jpeg",
        image_width=32,
        image_height=32,
        mask_data=_mask_bytes(),
        mask_mime="image/png",
        mask_width=16,
        mask_height=16,
        prompt="test",
    )
    with pytest.raises(EditValidationError):
        asyncio.run(provider.submit(request))
    asyncio.run(provider.aclose())


def test_base64_roundtrip() -> None:
    data = _image_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    assert base64.b64decode(encoded) == data
