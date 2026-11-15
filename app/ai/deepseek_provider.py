"""DeepSeek 视觉模型 Provider。

DeepSeek 官方 API 采用 OpenAI 兼容格式。根据 2026 年 11 月官方文档，
``deepseek-flash`` 支持图片输入，且支持 OpenAI 格式的 ``image_url``
（base64 data URL）与 JSON Output。``deepseek-v4-pro`` 不支持图片输入，
因此不能作为视觉分析模型。

官方文档：
- https://api-docs.deepseek.com/guides/vision
- https://api-docs.deepseek.com/guides/json_mode
"""

from __future__ import annotations

from typing import Any

import httpx

from app.ai.base import ImageInput, ModelResult
from app.ai.openai_compatible import OpenAICompatibleVisionProvider
from app.core.config import Settings
from app.core.exceptions import AIModelNotSupportVisionError

DEEPSEEK_VISION_MODELS = frozenset(
    {
        "deepseek-flash",
        "deepseek-v4-flash-vision-exp",
    }
)


class DeepSeekProvider(OpenAICompatibleVisionProvider):
    """DeepSeek 视觉模型调用实现。"""

    provider_name = "deepseek"

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        supported = settings.ai_model_supports_vision
        if supported is None:
            supported = settings.ai_model in DEEPSEEK_VISION_MODELS

        super().__init__(settings, transport=transport, provider_name="deepseek")
        self.supports_vision = supported

    async def analyze_image(
        self,
        image: ImageInput,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> ModelResult:
        if not self.supports_vision:
            raise AIModelNotSupportVisionError(
                detail=(
                    f"模型 {self.model} 不支持图片输入。"
                    "请使用 DeepSeek 视觉模型 deepseek-flash，"
                    "或通过 AI_MODEL_SUPPORTS_VISION=true 显式声明模型能力"
                )
            )
        return await super().analyze_image(
            image,
            prompt,
            system_prompt=system_prompt,
        )

    def _extra_payload(self) -> dict[str, Any]:
        enabled = self._settings.ai_enable_thinking
        payload: dict[str, Any] = {
            "thinking": {"type": "enabled" if enabled else "disabled"}
        }
        if enabled:
            payload["reasoning_effort"] = self._settings.ai_reasoning_effort
        return payload
