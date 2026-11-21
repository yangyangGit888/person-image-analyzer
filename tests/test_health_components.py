"""分组件健康检查、request_id 与模型能力信息测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.helpers import make_image_bytes


def test_health_reports_components_without_secret(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["components"]["api"]["status"] == "ok"
    assert body["components"]["sqlite"]["status"] == "ok"
    assert body["components"]["storage"]["status"] == "ok"
    assert body["components"]["ai"]["status"] == "ok"
    assert body["components"]["worker"]["status"] == "disabled"
    assert "api_key" not in response.text.lower()
    assert "test-key" not in response.text


def test_health_ai_degraded_without_credentials(make_client) -> None:
    with make_client(ai_api_key="") as (client, _):
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ai_configured"] is False
    assert body["components"]["ai"]["status"] == "unavailable"
    assert body["components"]["sqlite"]["status"] == "ok"


def test_request_id_is_echoed_and_invalid_value_is_replaced(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "req-123"})
    assert response.headers["X-Request-ID"] == "req-123"

    invalid = client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
    assert invalid.headers["X-Request-ID"] != "bad id with spaces"
    assert len(invalid.headers["X-Request-ID"]) > 10


def test_error_response_contains_request_id_and_details(client: TestClient) -> None:
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("broken.jpg", b"not-an-image", "image/jpeg")},
        headers={"X-Request-ID": "error-request-1"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["request_id"] == "error-request-1"
    assert body["error"]["code"] == "INVALID_IMAGE"
    assert response.headers["X-Request-ID"] == "error-request-1"


def test_validation_error_has_details(client: TestClient) -> None:
    response = client.post("/api/v1/analyze")

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert body["error"]["details"]
    assert body["request_id"]


def test_model_info_reports_prompt_versions(client: TestClient) -> None:
    response = client.get("/api/v1/model/info")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["vision_formats"] == ["jpeg", "png", "gif", "webp"]
    assert data["service_status"] == "configured"
    assert data["prompt_versions"]["person_analysis_system"] == "1"
    assert "test-key" not in response.text


def test_success_response_still_works_with_request_id(client: TestClient) -> None:
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("photo.png", make_image_bytes("PNG"), "image/png")},
        headers={"X-Request-ID": "success-request-1"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "success-request-1"
    assert response.json()["success"] is True
