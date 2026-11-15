"""视觉模型 Provider 工厂。"""

from __future__ import annotations

import httpx

from app.ai.base import BaseVisionProvider
from app.ai.deepseek_provider import DeepSeekProvider
from app.ai.openai_compatible import OpenAICompatibleVisionProvider
from app.core.config import Settings
from app.core.exceptions import AIConfigError

OPENAI_COMPATIBLE_PROVIDERS = frozenset({"openai", "gemini", "openai_compatible"})


def create_vision_provider(
    settings: Settings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> BaseVisionProvider:
    """根据配置创建视觉模型 Provider。

    - deepseek：使用官方验证过的视觉模型，并校验模型是否支持图片。
    - openai / gemini / openai_compatible：走 OpenAI 兼容接口，模型需自行配置。
    """

    provider = settings.ai_provider.lower()

    if provider == "deepseek":
        return DeepSeekProvider(settings, transport=transport)
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        return OpenAICompatibleVisionProvider(
            settings,
            transport=transport,
            provider_name=provider,
        )

    supported = ", ".join(sorted({"deepseek", *OPENAI_COMPATIBLE_PROVIDERS}))
    raise AIConfigError(
        detail=f"不支持的 AI_PROVIDER：{settings.ai_provider}；可选值：{supported}",
    )
