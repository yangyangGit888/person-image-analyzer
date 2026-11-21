"""图片 V2 多阶段分析、遮挡、关系与左右脚测试。"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tests.helpers import make_image_bytes
from tests.v2_helpers import V2FakeProvider


def _upload(client: TestClient) -> dict[str, Any]:
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("photo.png", make_image_bytes("PNG"), "image/png")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True
    return body["data"]


def test_v2_normal_pants_and_feet(make_client) -> None:
    provider = V2FakeProvider("normal")
    with make_client(provider=provider) as (client, _):
        data = _upload(client)

    assert data["schema_version"] == 2
    assert data["status"] == "completed"
    assert data["person_count"] == 1
    assert data["person_orientation"] == "front"
    assert data["body_observability"]["left_foot"]["visibility"] == "clear"
    assert len(data["feet"]) == 2
    assert all(foot["barefoot"] == "not_barefoot" for foot in data["feet"])

    categories = {item["category"] for item in data["items"]}
    assert {"pants", "hosiery", "socks", "shoes"} <= categories
    assert len(provider.calls) == 4
    assert data["persons"][0]["lower_body"]["wearing_pants"] == "true"
    assert data["persons"][0]["feet"]["wearing_shoes"] == "true"
    assert data["persons"][0]["feet"]["barefoot"] == "false"

    for item in data["items"]:
        if item["geometry"] is not None:
            bbox = item["geometry"]["bbox"]
            assert bbox[0] >= 0 and bbox[1] >= 0
            assert bbox[2] <= data["image_info"]["width"]
            assert bbox[3] <= data["image_info"]["height"]


def test_long_skirt_keeps_legs_and_feet_not_visible(make_client) -> None:
    provider = V2FakeProvider("long_skirt")
    with make_client(provider=provider) as (client, _):
        data = _upload(client)

    assert len(provider.calls) == 2
    assert data["body_observability"]["left_leg"]["visibility"] == "not_visible"
    assert data["body_observability"]["left_foot"]["visibility"] == "not_visible"
    assert data["body_observability"]["left_foot"]["occlusion_reason"] == [
        "long_skirt"
    ]
    for category in ("hosiery", "socks", "shoes"):
        matching = [item for item in data["items"] if item["category"] == category]
        assert matching
        assert all(item["presence"] == "not_visible" for item in matching)
        assert all("long_skirt" in item["occlusion_reason"] for item in matching)
    assert all(foot["barefoot"] == "not_visible" for foot in data["feet"])
    assert data["persons"][0]["feet"]["wearing_socks"] == "unknown"
    assert data["persons"][0]["feet"]["barefoot"] == "unknown"


def test_hosiery_socks_and_shoes_can_coexist(make_client) -> None:
    provider = V2FakeProvider("hosiery_socks_shoes")
    with make_client(provider=provider) as (client, _):
        data = _upload(client)

    hosiery = [item for item in data["items"] if item["category"] == "hosiery"]
    socks = [item for item in data["items"] if item["category"] == "socks"]
    shoes = [item for item in data["items"] if item["category"] == "shoes"]

    assert any(item["type"] == "pantyhose" for item in hosiery)
    assert all(item["presence"] == "detected" for item in hosiery)
    assert all(item["presence"] == "detected" for item in socks)
    assert all(item["presence"] == "detected" for item in shoes)
    assert data["persons"][0]["legs"]["wearing_tights"] == "true"
    assert data["persons"][0]["feet"]["wearing_socks"] == "true"
    assert data["persons"][0]["feet"]["wearing_shoes"] == "true"
    assert data["persons"][0]["feet"]["barefoot"] == "false"

    relations = {
        (relation["type"], relation["side"]): relation["status"]
        for relation in data["wearing_relations"]
    }
    assert relations[("hosiery_covers_feet", "left")] == "present"
    assert relations[("regular_socks_over_hosiery", "left")] == "present"
    assert relations[("shoes_over_socks", "left")] == "present"
    assert relations[("shoes_over_hosiery", "left")] == "present"


def test_one_shoe_and_one_bare_foot_are_independent(make_client) -> None:
    provider = V2FakeProvider("one_shoe")
    with make_client(provider=provider) as (client, _):
        data = _upload(client)

    shoes = {
        item["side"]: item
        for item in data["items"]
        if item["category"] == "shoes"
    }
    assert shoes["left"]["presence"] == "detected"
    assert shoes["right"]["presence"] == "not_detected"

    feet = {foot["side"]: foot for foot in data["feet"]}
    assert feet["left"]["barefoot"] == "not_barefoot"
    assert feet["right"]["barefoot"] == "barefoot"
    assert data["persons"][0]["feet"]["wearing_shoes"] == "true"


def test_feet_not_visible_does_not_become_barefoot(make_client) -> None:
    provider = V2FakeProvider("partial_feet")
    with make_client(provider=provider) as (client, _):
        data = _upload(client)

    assert len(provider.calls) == 3
    assert data["body_observability"]["left_foot"]["visibility"] == "not_visible"
    hosiery = [item for item in data["items"] if item["category"] == "hosiery"]
    assert any(item["type"] == "thigh_high" for item in hosiery)
    for category in ("socks", "shoes"):
        matching = [item for item in data["items"] if item["category"] == category]
        assert all(item["presence"] == "not_visible" for item in matching)
    assert all(foot["barefoot"] == "not_visible" for foot in data["feet"])


def test_multi_person_uses_legacy_compatibility_path(make_client) -> None:
    provider = V2FakeProvider("multi")
    with make_client(provider=provider) as (client, _):
        data = _upload(client)

    assert len(provider.calls) == 2
    assert data["status"] == "unsupported_multi_person"
    assert len(data["persons"]) == 2
    assert data["items"] == []
    assert any(
        warning["code"] == "MULTI_PERSON_V2_UNSUPPORTED"
        for warning in data["warnings"]
    )


def test_side_unknown_is_not_guessed(make_client) -> None:
    provider = V2FakeProvider("side_unknown")
    with make_client(provider=provider) as (client, _):
        data = _upload(client)

    assert data["side_assignment_confidence"] == "low"
    assert data["side_basis"] == "unknown"
    assert any(
        warning["code"] == "SIDE_ASSIGNMENT_UNKNOWN"
        for warning in data["warnings"]
    )
    for category in ("hosiery", "socks", "shoes"):
        matching = [item for item in data["items"] if item["category"] == category]
        assert matching
        assert all(item["side"] == "unknown" for item in matching)
        assert {item["image_side"] for item in matching} == {"left", "right"}
    assert {foot["side"] for foot in data["feet"]} == {"unknown"}
    assert {foot["image_side"] for foot in data["feet"]} == {"left", "right"}


def test_detail_budget_is_honored(make_client) -> None:
    provider = V2FakeProvider("normal")
    with make_client(provider=provider, ai_max_calls_per_analysis=2) as (client, _):
        data = _upload(client)

    assert len(provider.calls) == 2
    assert data["status"] == "completed"
    assert any(
        warning["code"] == "ANALYSIS_LIMITATION"
        for warning in data["warnings"]
    )
    assert all(
        foot["barefoot"] in {"unknown", "not_visible"} for foot in data["feet"]
    )
