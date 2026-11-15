"""健康检查与模型配置接口测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_check(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["ai_configured"] is True
    assert "test-key" not in response.text
    assert "api_key" not in response.text.lower()


def test_health_without_api_key(make_client) -> None:
    with make_client(ai_api_key="") as (client, _):
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ai_configured"] is False


def test_model_info_does_not_expose_key(client: TestClient) -> None:
    response = client.get("/api/v1/model/info")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["provider"] == "deepseek"
    assert body["data"]["model"] == "deepseek-flash"
    assert body["data"]["base_url"] == "https://api.deepseek.com"
    assert body["data"]["supports_vision"] is True
    assert body["data"]["api_key_configured"] is True
    assert "test-key" not in response.text
    assert "api_key" not in body["data"]


def test_root_endpoint(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"
