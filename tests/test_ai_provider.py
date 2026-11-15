"""使用 MockTransport 测试 DeepSeek Provider，不产生真实 API 费用。"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from app.ai.base import ImageInput
from app.ai.deepseek_provider import DeepSeekProvider
from app.ai.parsing import parse_json_content
from app.core.config import Settings
from app.core.exceptions import (
    AIAuthenticationError,
    AIConfigError,
    AIInvalidJSONError,
    AIModelNotSupportVisionError,
    AITimeoutError,
)

IMAGE = ImageInput(
    data=b"fake-jpeg-bytes",
    mime_type="image/jpeg",
    width=640,
    height=480,
)


def _settings(**overrides: Any) -> Settings:
    config: dict[str, Any] = {
        "ai_api_key": "unit-test-secret-key",
        "ai_provider": "deepseek",
        "ai_model": "deepseek-flash",
        "ai_base_url": "https://api.deepseek.com",
        "ai_max_retries": 0,
        "ai_retry_backoff": 0,
        "ai_enable_thinking": False,
        "ai_timeout": 5,
    }
    config.update(overrides)
    return Settings(_env_file=None, **config)


def _success_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "model": "deepseek-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1100, "completion_tokens": 300},
        },
    )


def _run(provider: DeepSeekProvider) -> Any:
    async def _inner() -> Any:
        try:
            return await provider.analyze_image(
                IMAGE,
                "请分析人物并返回 json",
                system_prompt="你是视觉分析专家，只返回 json",
            )
        finally:
            await provider.aclose()

    return asyncio.run(_inner())


def test_deepseek_request_format_and_parsing() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        captured["payload"] = json.loads(request.content)
        content = json.dumps(
            {
                "person_count": 1,
                "persons": [{"person_id": 1, "feet": {"wearing_socks": "unknown"}}],
            },
            ensure_ascii=False,
        )
        return _success_response(content)

    provider = DeepSeekProvider(_settings(), transport=httpx.MockTransport(handler))
    result = _run(provider)

    payload = captured["payload"]
    assert captured["headers"]["authorization"] == "Bearer unit-test-secret-key"
    assert payload["model"] == "deepseek-flash"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in payload
    assert payload["messages"][0]["role"] == "system"

    user_content = payload["messages"][1]["content"]
    assert user_content[0]["type"] == "text"
    image_part = user_content[1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert image_part["image_url"]["detail"] == "high"

    assert result.parsed["person_count"] == 1
    assert result.usage["prompt_tokens"] == 1100
    assert result.request_id == "chatcmpl-test"


def test_connect_ip_keeps_host_and_sni() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = request.url
        captured["host"] = request.headers.get("host")
        captured["sni"] = request.extensions.get("sni_hostname")
        return _success_response('{"person_count": 0, "persons": []}')

    provider = DeepSeekProvider(
        _settings(ai_connect_ip="120.221.61.47"),
        transport=httpx.MockTransport(handler),
    )
    _run(provider)

    assert captured["url"].host == "120.221.61.47"
    assert captured["url"].path == "/chat/completions"
    assert captured["host"] == "api.deepseek.com"
    assert captured["sni"] == "api.deepseek.com"


def test_retry_on_server_error() -> None:
    calls = {"count": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, text="temporarily unavailable")
        return _success_response('{"person_count": 0, "persons": []}')

    provider = DeepSeekProvider(
        _settings(ai_max_retries=1),
        transport=httpx.MockTransport(handler),
    )
    result = _run(provider)

    assert calls["count"] == 2
    assert result.parsed["persons"] == []


def test_auth_error_mapping() -> None:
    provider = DeepSeekProvider(
        _settings(),
        transport=httpx.MockTransport(
            lambda _: httpx.Response(401, text="invalid api key")
        ),
    )

    with pytest.raises(AIAuthenticationError) as exc_info:
        _run(provider)

    assert "unit-test-secret-key" not in str(exc_info.value)


def test_invalid_json_response_mapping() -> None:
    provider = DeepSeekProvider(
        _settings(),
        transport=httpx.MockTransport(
            lambda _: _success_response("这不是 json")
        ),
    )

    with pytest.raises(AIInvalidJSONError):
        _run(provider)


def test_empty_content_response_mapping() -> None:
    provider = DeepSeekProvider(
        _settings(),
        transport=httpx.MockTransport(lambda _: _success_response("   ")),
    )

    with pytest.raises(Exception) as exc_info:
        _run(provider)

    assert getattr(exc_info.value, "code", None) == "AI_EMPTY_RESPONSE"


def test_model_not_support_vision_mapping() -> None:
    provider = DeepSeekProvider(
        _settings(),
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                400,
                text="this model does not support image input",
            )
        ),
    )

    with pytest.raises(AIModelNotSupportVisionError):
        _run(provider)


def test_timeout_mapping_with_retry() -> None:
    calls = {"count": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise httpx.ReadTimeout("timed out")

    provider = DeepSeekProvider(
        _settings(ai_max_retries=1),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(AITimeoutError):
        _run(provider)

    assert calls["count"] == 2


def test_missing_api_key_raises_config_error() -> None:
    provider = DeepSeekProvider(
        _settings(ai_api_key=""),
        transport=httpx.MockTransport(
            lambda _: _success_response('{"person_count": 0, "persons": []}')
        ),
    )

    with pytest.raises(AIConfigError):
        _run(provider)


def test_text_only_model_is_rejected_before_request() -> None:
    provider = DeepSeekProvider(_settings(ai_model="deepseek-v4-pro"))

    assert provider.supports_vision is False
    with pytest.raises(AIModelNotSupportVisionError) as exc_info:
        _run(provider)

    assert "不支持图片输入" in str(exc_info.value)


def test_parse_json_content_strips_extra_whitespace() -> None:
    assert parse_json_content('  {"a": 1}  ') == {"a": 1}
