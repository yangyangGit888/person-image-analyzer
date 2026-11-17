"""阿里云百炼万相 2.1 局部图像编辑 Provider。

接口事实以官方文档为准：
https://help.aliyun.com/zh/model-studio/wanx-image-edit-api-reference
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import httpx

from app.ai.image_edit.base import (
    BaseImageEditProvider,
    ImageEditRequest,
    ImageEditTask,
)
from app.core.config import Settings
from app.core.exceptions import (
    EditValidationError,
    ImageEditAuthError,
    ImageEditConfigError,
    ImageEditContentRejectedError,
    ImageEditFailedError,
    ImageEditInvalidOutputError,
    ImageEditRateLimitError,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class WanxImageEditProvider(BaseImageEditProvider):
    provider_name = "wanx"

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self.model = settings.image_edit_model
        if settings.dashscope_workspace_id:
            base = (
                "https://"
                f"{settings.dashscope_workspace_id}.cn-beijing.maas.aliyuncs.com/api/v1"
            )
        else:
            base = settings.dashscope_base_url.rstrip("/")
        self.base_url = base
        self._submit_url = f"{base}/services/aigc/image2image/image-synthesis"
        self._task_url = f"{base}/tasks"
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                settings.image_edit_timeout,
                connect=min(15.0, settings.image_edit_timeout),
            ),
            transport=transport,
            follow_redirects=False,
        )

    def _headers(self) -> dict[str, str]:
        api_key = self._settings.dashscope_api_key_value
        if not api_key:
            raise ImageEditConfigError(
                detail="未配置 DASHSCOPE_API_KEY，无法调用万相图像编辑服务",
            )
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }

    async def submit(self, request: ImageEditRequest) -> ImageEditTask:
        if (request.image_width, request.image_height) != (
            request.mask_width,
            request.mask_height,
        ):
            raise EditValidationError(detail="Mask 尺寸必须与原图完全一致")
        headers = self._headers()
        payload: dict[str, Any] = {
            "model": self.model,
            "input": {
                "function": self._settings.image_edit_function,
                "prompt": request.prompt,
                "base_image_url": self._data_url(
                    request.image_data, request.image_mime
                ),
                "mask_image_url": self._data_url(
                    request.mask_data, request.mask_mime
                ),
            },
            "parameters": {
                "n": 1,
                "watermark": self._settings.image_edit_watermark,
            },
        }
        if request.seed is not None:
            payload["parameters"]["seed"] = request.seed

        total_attempts = self._settings.image_edit_max_retries + 1
        for attempt in range(total_attempts):
            try:
                response = await self._client.post(
                    self._submit_url, json=payload, headers=headers
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt < total_attempts - 1:
                    await self._sleep(attempt)
                    continue
                raise ImageEditFailedError(
                    detail="无法连接万相服务，请检查网络或 DASHSCOPE_BASE_URL",
                ) from exc

            if response.status_code in _RETRYABLE_STATUS and attempt < total_attempts - 1:
                logger.warning(
                    "万相提交失败，准备重试: status=%s attempt=%s/%s",
                    response.status_code,
                    attempt + 1,
                    total_attempts,
                )
                await self._sleep(attempt)
                continue
            if response.status_code not in {200, 202}:
                self._raise_for_status(response)

            data = self._parse_json(response)
            output = data.get("output")
            if not isinstance(output, dict):
                raise ImageEditInvalidOutputError(detail="万相响应缺少 output")
            task_id = output.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise ImageEditInvalidOutputError(detail="万相响应缺少 task_id")
            return ImageEditTask(
                task_id=task_id,
                status=str(output.get("task_status") or "PENDING").upper(),
                raw=data,
            )

        raise ImageEditFailedError(detail="万相任务提交失败")

    async def fetch(self, task_id: str) -> ImageEditTask:
        headers = self._headers()
        total_attempts = self._settings.image_edit_max_retries + 1
        last_response: httpx.Response | None = None
        for attempt in range(total_attempts):
            try:
                response = await self._client.get(
                    f"{self._task_url}/{task_id}", headers=headers
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt < total_attempts - 1:
                    await self._sleep(attempt)
                    continue
                raise ImageEditFailedError(
                    detail="查询万相任务状态失败，请稍后重试",
                ) from exc

            last_response = response
            if response.status_code in _RETRYABLE_STATUS and attempt < total_attempts - 1:
                await self._sleep(attempt)
                continue
            if response.status_code != 200:
                self._raise_for_status(response)

            data = self._parse_json(response)
            output = data.get("output")
            if not isinstance(output, dict):
                raise ImageEditInvalidOutputError(detail="万相任务响应缺少 output")
            status = str(output.get("task_status") or "UNKNOWN").upper()
            urls: list[str] = []
            results = output.get("results")
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict) and isinstance(item.get("url"), str):
                        urls.append(item["url"])
            error_code = output.get("code")
            error_message = output.get("message")
            return ImageEditTask(
                task_id=str(output.get("task_id") or task_id),
                status=status,
                result_urls=urls,
                error_code=str(error_code) if error_code else None,
                error_message=str(error_message) if error_message else None,
                raw=data,
            )

        if last_response is not None:
            self._raise_for_status(last_response)
        raise ImageEditFailedError(detail="查询万相任务状态失败")

    async def download(self, url: str) -> bytes:
        total_attempts = self._settings.image_edit_max_retries + 1
        for attempt in range(total_attempts):
            try:
                response = await self._client.get(url)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt < total_attempts - 1:
                    await self._sleep(attempt)
                    continue
                raise ImageEditFailedError(
                    detail="下载万相编辑结果失败",
                ) from exc
            if response.status_code in _RETRYABLE_STATUS and attempt < total_attempts - 1:
                await self._sleep(attempt)
                continue
            if response.status_code != 200:
                raise ImageEditFailedError(
                    detail=f"下载万相编辑结果失败（HTTP {response.status_code}）",
                )
            content_type = response.headers.get("content-type", "")
            if content_type and not content_type.startswith("image/"):
                raise ImageEditInvalidOutputError(
                    detail="万相返回的结果不是图片",
                )
            data = response.content
            if not data:
                raise ImageEditInvalidOutputError(detail="万相返回的图片为空")
            if len(data) > 30 * 1024 * 1024:
                raise ImageEditInvalidOutputError(detail="万相返回的图片过大")
            return data
        raise ImageEditFailedError(detail="下载万相编辑结果失败")

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        body = self._safe_text(response.text)
        logger.warning("万相 API 错误: status=%s body=%s", status, body)
        if status in {401, 403}:
            raise ImageEditAuthError()
        if status == 429:
            raise ImageEditRateLimitError()
        if status == 400:
            lowered = response.text.lower()
            if "datainspectionfailed" in lowered or "content" in lowered:
                raise ImageEditContentRejectedError()
            raise ImageEditFailedError(detail="万相请求参数被拒绝，请检查任务配置")
        if status >= 500:
            raise ImageEditFailedError(detail="万相服务暂时不可用，请稍后重试")
        raise ImageEditFailedError(detail=f"万相服务调用失败（HTTP {status}）")

    @staticmethod
    def _parse_json(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise ImageEditInvalidOutputError(
                detail="万相返回了无法解析的响应",
            ) from exc
        if not isinstance(data, dict):
            raise ImageEditInvalidOutputError(detail="万相响应不是 JSON 对象")
        return data

    @staticmethod
    def _data_url(data: bytes, mime_type: str) -> str:
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    async def _sleep(self, attempt: int) -> None:
        delay = min(8.0, 0.8 * (2**attempt))
        if delay > 0:
            await asyncio.sleep(delay)

    def _safe_text(self, text: str, limit: int = 300) -> str:
        key = self._settings.dashscope_api_key_value
        safe = text.replace(key, "***") if key else text
        return safe[:limit].replace("\n", " ")

    async def aclose(self) -> None:
        await self._client.aclose()
