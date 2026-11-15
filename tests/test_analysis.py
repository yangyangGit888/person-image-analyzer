"""AI 分析流程、结果归一化与异常处理测试。"""

from __future__ import annotations

import pytest

from app.ai.parsing import parse_json_content
from app.core.exceptions import (
    AIAPIError,
    AIAuthenticationError,
    AIInvalidJSONError,
    AITimeoutError,
)
from tests.helpers import FakeProvider, make_image_bytes, two_person_result


def _upload(client):
    return client.post(
        "/api/v1/analyze",
        files={"file": ("photo.png", make_image_bytes("PNG"), "image/png")},
    )


def test_multi_person_results_are_independent(make_client) -> None:
    provider = FakeProvider(two_person_result())
    with make_client(provider=provider) as (client, _):
        response = _upload(client)

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["image_info"]["person_count"] == 2
    assert [person["person_id"] for person in data["persons"]] == [1, 2]

    first, second = data["persons"]
    assert first["upper_body"]["top_color"] == "红色"
    assert first["feet"]["wearing_socks"] == "true"
    assert first["feet"]["shoes_color"] == "白色"

    assert second["upper_body"]["top_color"] == "蓝色"
    assert second["lower_body"]["wearing_skirt"] == "true"
    assert second["feet"]["wearing_socks"] == "unknown"
    assert second["evidence"]["feet"] == "脚部在画面外"


def test_tristate_and_missing_fields_normalization(make_client) -> None:
    raw = {
        "person_count": 1,
        "persons": [
            {
                "person_id": 99,
                "head": {"wearing_hat": True, "hat_type": "", "hat_color": "红色"},
                "upper_body": {"wearing_top": 1, "wearing_jacket": "NO"},
                "feet": {"wearing_socks": False},
                "confidence": {"head": "0.8", "feet": 1.7},
                "evidence": {"head": "帽子清晰可见"},
            }
        ],
    }
    with make_client(provider=FakeProvider(raw)) as (client, _):
        response = _upload(client)

    assert response.status_code == 200
    person = response.json()["data"]["persons"][0]
    assert person["person_id"] == 1
    assert person["head"]["wearing_hat"] == "true"
    assert person["head"]["hat_type"] == "unknown"
    assert person["head"]["hat_color"] == "红色"
    assert person["head"]["wearing_glasses"] == "unknown"
    assert person["upper_body"]["wearing_top"] == "true"
    assert person["upper_body"]["wearing_jacket"] == "false"
    assert person["upper_body"]["top_color"] == "unknown"
    assert person["feet"]["wearing_socks"] == "false"
    assert person["feet"]["socks_type"] == "unknown"
    assert person["legs"]["skin_visible"] == "unknown"
    assert person["visibility"]["full_body_visible"] == "unknown"
    assert person["confidence"]["head"] == 0.8
    assert person["confidence"]["feet"] == 1.0
    assert person["confidence"]["legs"] is None
    assert person["position"]["bbox"] is None


def test_no_person_detected(make_client) -> None:
    with make_client(provider=FakeProvider({"person_count": 0, "persons": []})) as (
        client,
        _,
    ):
        response = _upload(client)

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "NO_PERSON_DETECTED"


def test_missing_persons_field_is_invalid_response(make_client) -> None:
    with make_client(provider=FakeProvider({"person_count": 1})) as (client, _):
        response = _upload(client)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "AI_INVALID_RESPONSE"


def test_ai_timeout_maps_to_504(make_client) -> None:
    with make_client(provider=FakeProvider(error=AITimeoutError())) as (client, _):
        response = _upload(client)

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "AI_TIMEOUT"


def test_ai_auth_error_does_not_leak_key(make_client) -> None:
    with make_client(provider=FakeProvider(error=AIAuthenticationError())) as (
        client,
        _,
    ):
        response = _upload(client)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "AI_AUTH_ERROR"
    assert "test-key" not in response.text


def test_invalid_json_from_model(make_client) -> None:
    with make_client(provider=FakeProvider(error=AIInvalidJSONError())) as (client, _):
        response = _upload(client)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "AI_INVALID_JSON"


def test_ai_api_error_envelope(make_client) -> None:
    with make_client(provider=FakeProvider(error=AIAPIError())) as (client, _):
        response = _upload(client)

    assert response.status_code == 502
    body = response.json()
    assert body["success"] is False
    assert body["message"] == "图片分析失败"
    assert body["error"] == {
        "code": "AI_API_ERROR",
        "detail": "模型调用失败",
    }


def test_parse_json_content_rejects_invalid_text() -> None:
    with pytest.raises(AIInvalidJSONError):
        parse_json_content("抱歉，我无法识别这张图片")


def test_parse_json_content_rejects_non_object() -> None:
    with pytest.raises(AIInvalidJSONError):
        parse_json_content("[1, 2, 3]")


def test_parse_json_content_accepts_fenced_json() -> None:
    parsed = parse_json_content('```json\n{"person_count": 0, "persons": []}\n```')

    assert parsed["person_count"] == 0
