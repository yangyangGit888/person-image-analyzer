"""OpenAI 兼容格式的视觉模型 Provider。

DeepSeek、OpenAI 以及 Gemini 的 OpenAI 兼容端点均使用
``/chat/completions`` + ``image_url`` 的请求结构，因此可以复用同一实现。
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from app.ai.base import BaseVisionProvider, ImageInput, ModelResult
from app.ai.parsing import parse_json_content
from app.core.config import Settings
from app.core.exceptions import (
    AIAPIError,
    AIAuthenticationError,
    AIConfigError,
    AIEmptyResponseError,
    AIModelNotSupportVisionError,
    AIRateLimitError,
    AIServiceUnavailableError,
    AITimeoutError,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleVisionProvider(BaseVisionProvider):
    """通过 OpenAI 兼容的 Chat Completions 接口调用视觉模型。"""

    provider_name = "openai_compatible"

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        provider_name: str | None = None,
    ) -> None:
        self._settings = settings
        self.model = settings.ai_model
        self.base_url = settings.ai_base_url
        self.supports_vision = True
        if provider_name:
            self.provider_name = provider_name

        self._endpoint = self._build_endpoint(settings.ai_base_url)
        self._request_extensions: dict[str, Any] = {}
        self._host_header: str | None = None
        if settings.ai_connect_ip:
            self._apply_connect_ip(settings.ai_connect_ip)

        timeout = httpx.Timeout(
            settings.ai_timeout,
            connect=min(10.0, settings.ai_timeout),
        )
        self._client = httpx.AsyncClient(
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
            headers={"Accept": "application/json"},
        )

    def _apply_connect_ip(self, connect_ip: str) -> None:
        """把请求地址的 host 换成指定 IP，同时保留域名 Host 与 TLS SNI。

        该选项用于本地 DNS 返回的节点不可达、但该域名在其它节点可用的场景。
        TLS 证书仍按原域名校验，不会关闭证书验证。
        """

        parsed = urlsplit(self._endpoint)
        host = parsed.hostname
        if not host:
            raise AIConfigError(detail="AI_BASE_URL 缺少合法域名")

        default_port = 443 if parsed.scheme == "https" else 80
        port = parsed.port or default_port
        ip_host = f"[{connect_ip}]" if ":" in connect_ip else connect_ip
        netloc = ip_host
        if port != default_port:
            netloc = f"{ip_host}:{port}"

        self._endpoint = urlunsplit(
            (parsed.scheme, netloc, parsed.path, parsed.query, "")
        )
        self._host_header = host if port == default_port else f"{host}:{port}"
        if parsed.scheme == "https":
            self._request_extensions["sni_hostname"] = host

    @staticmethod
    def _build_endpoint(base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _build_payload(
        self,
        image: ImageInput,
        prompt: str,
        system_prompt: str | None,
    ) -> dict[str, Any]:
        encoded = base64.b64encode(image.data).decode("ascii")
        mime_type = image.mime_type or "image/jpeg"
        data_url = f"data:{mime_type};base64,{encoded}"

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                            "detail": self._settings.ai_vision_detail,
                        },
                    },
                ],
            }
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_tokens": self._settings.ai_max_output_tokens,
            "stream": False,
        }
        payload.update(self._extra_payload())
        return payload

    def _extra_payload(self) -> dict[str, Any]:
        """不同厂商的扩展参数钩子。"""

        return {}

    def _headers(self) -> dict[str, str]:
        api_key = self._settings.api_key
        if not api_key:
            raise AIConfigError(
                detail="未配置 AI_API_KEY，无法调用模型服务",
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self._host_header:
            headers["Host"] = self._host_header
        return headers

    async def analyze_image(
        self,
        image: ImageInput,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> ModelResult:
        headers = self._headers()
        payload = self._build_payload(image, prompt, system_prompt)

        total_attempts = self._settings.ai_max_retries + 1
        last_error: Exception | None = None

        for attempt in range(total_attempts):
            try:
                response = await self._client.post(
                    self._endpoint,
                    json=payload,
                    headers=headers,
                    extensions=self._request_extensions or None,
                )
            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning(
                    "模型请求超时: attempt=%s/%s",
                    attempt + 1,
                    total_attempts,
                )
                if attempt < total_attempts - 1:
                    await self._sleep_before_retry(attempt)
                    continue
                raise AITimeoutError() from exc
            except httpx.TransportError as exc:
                last_error = exc
                logger.warning(
                    "模型服务连接失败: attempt=%s/%s type=%s",
                    attempt + 1,
                    total_attempts,
                    type(exc).__name__,
                )
                if attempt < total_attempts - 1:
                    await self._sleep_before_retry(attempt)
                    continue
                raise AIAPIError(
                    detail="无法连接模型服务，请检查 AI_BASE_URL 或网络",
                ) from exc

            if response.status_code == 200:
                return self._parse_response(response)

            if (
                response.status_code in {408, 429, 500, 502, 503, 504}
                and attempt < total_attempts - 1
            ):
                logger.warning(
                    "模型请求失败，准备重试: status=%s attempt=%s/%s",
                    response.status_code,
                    attempt + 1,
                    total_attempts,
                )
                await self._sleep_before_retry(attempt, response=response)
                continue

            self._raise_for_status(response)

        raise AIAPIError() from last_error

    async def _sleep_before_retry(
        self,
        attempt: int,
        *,
        response: httpx.Response | None = None,
    ) -> None:
        delay = self._settings.ai_retry_backoff * (2**attempt)
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                with contextlib.suppress(ValueError):
                    delay = min(float(retry_after), 10.0)
        if delay > 0:
            await asyncio.sleep(delay)

    def _parse_response(self, response: httpx.Response) -> ModelResult:
        try:
            data = response.json()
        except ValueError as exc:
            raise AIAPIError(
                detail="模型服务返回了无法解析的响应",
            ) from exc

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AIAPIError(detail="模型服务响应结构异常")

        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message")
        content: Any = message.get("content") if isinstance(message, dict) else None

        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            content = "".join(parts)

        if not isinstance(content, str) or not content.strip():
            finish_reason = first.get("finish_reason")
            logger.warning("模型返回空内容: finish_reason=%s", finish_reason)
            raise AIEmptyResponseError(
                detail=(
                    "模型返回内容为空或输出被截断，请重试"
                    if finish_reason == "length"
                    else "模型返回内容为空，请重试"
                )
            )

        parsed = parse_json_content(content)
        usage_raw = data.get("usage")
        usage: dict[str, int] = {}
        if isinstance(usage_raw, dict):
            for key, value in usage_raw.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    usage[str(key)] = value

        request_id = data.get("id")
        return ModelResult(
            content=content,
            parsed=parsed,
            model=str(data.get("model") or self.model),
            provider=self.provider_name,
            usage=usage,
            request_id=request_id if isinstance(request_id, str) else None,
        )

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        body = self._safe_log_text(response.text)
        logger.warning(
            "模型服务返回错误: status=%s provider=%s model=%s body=%s",
            status,
            self.provider_name,
            self.model,
            body,
        )

        if status in {401, 403}:
            raise AIAuthenticationError()
        if status == 404:
            raise AIAPIError(
                detail="模型或接口地址不存在，请检查 AI_BASE_URL 与 AI_MODEL",
            )
        if status == 408:
            raise AITimeoutError()
        if status == 413:
            raise AIAPIError(detail="图片数据过大，模型服务拒绝接收")
        if status == 429:
            raise AIRateLimitError()
        if status == 400:
            lowered = response.text.lower()
            if any(
                keyword in lowered
                for keyword in ("image", "vision", "multimodal", "modality")
            ):
                raise AIModelNotSupportVisionError()
            raise AIAPIError(detail="模型拒绝请求，请检查模型名称和请求参数")
        if status >= 500:
            raise AIServiceUnavailableError()
        raise AIAPIError(detail=f"模型服务调用失败（HTTP {status}）")

    def _safe_log_text(self, text: str, limit: int = 300) -> str:
        api_key = self._settings.api_key
        safe = text.replace(api_key, "***") if api_key else text
        return safe[:limit].replace("\n", " ")

    async def aclose(self) -> None:
        await self._client.aclose()
