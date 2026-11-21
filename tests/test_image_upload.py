"""图片上传与校验接口测试。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.helpers import make_image_bytes


@pytest.mark.parametrize(
    ("fmt", "filename", "content_type"),
    [
        ("JPEG", "photo.jpg", "image/jpeg"),
        ("PNG", "photo.png", "image/png"),
        ("WEBP", "photo.webp", "image/webp"),
        ("HEIF", "photo.heic", "image/heic"),
    ],
)
def test_upload_supported_formats(
    client: TestClient,
    fmt: str,
    filename: str,
    content_type: str,
) -> None:
    response = client.post(
        "/api/v1/analyze",
        files={"file": (filename, make_image_bytes(fmt), content_type)},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["message"] == "分析成功"
    assert body["data"]["image_info"]["width"] == 640
    assert body["data"]["image_info"]["height"] == 480
    assert body["data"]["image_info"]["person_count"] == 1
    assert body["data"]["persons"][0]["person_id"] == 1
    assert body["data"]["persons"][0]["head"]["wearing_hat"] == "true"
    assert body["data"]["persons"][0]["feet"]["wearing_socks"] == "unknown"
    assert "test-key" not in response.text


def test_content_detection_not_extension(client: TestClient) -> None:
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("photo.txt", make_image_bytes("PNG"), "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["success"] is True


def test_heic_content_with_png_extension(client: TestClient) -> None:
    response = client.post(
        "/api/v1/analyze",
        files={
            "file": (
                "iphone-photo.png",
                make_image_bytes("HEIF"),
                "image/png",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["success"] is True


def test_accept_gif_first_frame(client: TestClient) -> None:
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("animation.gif", make_image_bytes("GIF"), "image/gif")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["image_info"]["width"] == 640


def test_reject_unsupported_format(client: TestClient) -> None:
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("bitmap.bmp", make_image_bytes("BMP"), "image/bmp")},
    )

    assert response.status_code == 415
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "UNSUPPORTED_IMAGE_FORMAT"
    assert "test-key" not in response.text


def test_reject_corrupted_image(client: TestClient) -> None:
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("broken.jpg", b"this is not an image", "image/jpeg")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "INVALID_IMAGE"


def test_reject_empty_file(client: TestClient) -> None:
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("empty.png", b"", "image/png")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "EMPTY_FILE"


def test_reject_too_large_image(make_client) -> None:
    with make_client(max_upload_size_mb=1) as (client, _):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("big.jpg", b"\x00" * (1024 * 1024 + 100), "image/jpeg")},
        )

    assert response.status_code == 413
    body = response.json()
    assert body["error"]["code"] == "IMAGE_TOO_LARGE"


def test_reject_huge_request_before_parsing(make_client) -> None:
    with make_client(max_upload_size_mb=1) as (client, _):
        response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "huge.jpg",
                    b"\x00" * (3 * 1024 * 1024),
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_missing_file_returns_unified_error(client: TestClient) -> None:
    response = client.post("/api/v1/analyze")

    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "VALIDATION_ERROR"


def test_image_service_called_once(make_client) -> None:
    with make_client() as (client, provider):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("photo.png", make_image_bytes("PNG"), "image/png")},
        )
        assert response.status_code == 200
        assert len(provider.calls) == 1
        image = provider.calls[0]["image"]
        assert image.mime_type == "image/jpeg"
        assert image.data.startswith(b"\xff\xd8")
        assert "json" in provider.calls[0]["system_prompt"].lower()
