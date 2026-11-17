"""多人、生成失败与任务冲突测试。"""

from __future__ import annotations

import time

from tests.helpers import make_image_bytes, two_person_result


def _two_person_region_result() -> dict:
    def person_regions(index: int, top_bbox: dict, color: str) -> dict:
        return {
            "person_index": index,
            "regions": {
                "hat": {
                    "existence": "missing",
                    "bbox": None,
                    "confidence": 0.8,
                    "type": "unknown",
                    "color": "unknown",
                },
                "glasses": {
                    "existence": "missing",
                    "bbox": None,
                    "confidence": 0.8,
                    "type": "unknown",
                    "color": "unknown",
                },
                "hair": {
                    "existence": "existing",
                    "bbox": top_bbox,
                    "confidence": 0.8,
                    "type": "短发",
                    "color": "黑色",
                },
                "earring": {
                    "existence": "missing",
                    "bbox": None,
                    "confidence": None,
                    "type": "unknown",
                    "color": "unknown",
                },
                "top": {
                    "existence": "existing",
                    "bbox": top_bbox,
                    "confidence": 0.9,
                    "type": "上衣",
                    "color": color,
                },
                "pants": {
                    "existence": "existing",
                    "bbox": top_bbox,
                    "confidence": 0.9,
                    "type": "裤子",
                    "color": "黑色",
                },
                "shoes": {
                    "existence": "missing",
                    "bbox": None,
                    "confidence": None,
                    "type": "unknown",
                    "color": "unknown",
                },
            },
        }

    return {
        "persons": [
            person_regions(
                1,
                {"x": 0.08, "y": 0.25, "width": 0.32, "height": 0.3},
                "红色",
            ),
            person_regions(
                2,
                {"x": 0.58, "y": 0.25, "width": 0.32, "height": 0.3},
                "蓝色",
            ),
        ]
    }


def _two_person_with_bboxes():
    result = two_person_result()
    result["persons"][0]["position"] = {
        "description": "左侧",
        "bbox": {"x": 0.05, "y": 0.1, "width": 0.4, "height": 0.8},
    }
    result["persons"][1]["position"] = {
        "description": "右侧",
        "bbox": {"x": 0.55, "y": 0.1, "width": 0.4, "height": 0.8},
    }
    return result


def _upload(client):
    response = client.post(
        "/api/v1/edit/images",
        files={"file": ("group.png", make_image_bytes("PNG", (1000, 1000)), "image/png")},
    )
    assert response.status_code == 201, response.text
    return response.json()["data"]


def _analyze(client, image_id):
    response = client.post(f"/api/v1/edit/images/{image_id}/analyze")
    assert response.status_code == 200, response.text
    return response.json()["data"]


def test_multi_person_regions_are_not_mixed(make_edit_client) -> None:
    with make_edit_client(
        person_result=_two_person_with_bboxes(),
        region_result=_two_person_region_result(),
    ) as (client, _):
        image = _upload(client)
        data = _analyze(client, image["image_id"])

    assert len(data["persons"]) == 2
    assert data["persons"][0]["person_id"] != data["persons"][1]["person_id"]
    assert data["persons"][0]["ordinal"] == 1
    assert data["persons"][1]["ordinal"] == 2
    assert len(data["regions"]) == 14

    top_regions = [
        region for region in data["regions"] if region["region_type"] == "top"
    ]
    assert len(top_regions) == 2
    by_person = {region["person_id"]: region for region in top_regions}
    left = by_person[data["persons"][0]["person_id"]]
    right = by_person[data["persons"][1]["person_id"]]
    assert left["attributes"]["color"] == "红色"
    assert right["attributes"]["color"] == "蓝色"
    assert left["bbox"]["x"] < 500
    assert right["bbox"]["x"] > 500


def test_generation_fails_without_dashscope_key(make_edit_client) -> None:
    with make_edit_client(
        image_edit_provider="wanx",
        dashscope_api_key="",
    ) as (client, _):
        image = _upload(client)
        data = _analyze(client, image["image_id"])
        person = data["persons"][0]
        task_response = client.post(
            "/api/v1/edit/tasks",
            json={
                "image_id": image["image_id"],
                "person_id": person["person_id"],
                "region_type": "hat",
                "operation": "add",
                "style_id": "baseball_cap",
                "color_id": "black",
            },
        )
        assert task_response.status_code == 201, task_response.text
        task_id = task_response.json()["data"]["task_id"]

        response = client.post(
            "/api/v1/edit/generations",
            json={
                "image_id": image["image_id"],
                "person_id": person["person_id"],
                "task_ids": [task_id],
            },
        )
        assert response.status_code == 202, response.text
        generation_id = response.json()["data"]["generation"]["generation_id"]

        result = None
        for _ in range(40):
            detail = client.get(f"/api/v1/edit/generations/{generation_id}")
            result = detail.json()["data"]
            if result["generation"]["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)

    assert result is not None
    assert result["generation"]["status"] == "failed"
    assert result["generation"]["error_code"] == "IMAGE_EDIT_CONFIG_ERROR"
    assert result["steps"][0]["status"] == "failed"


def test_duplicate_task_for_same_region_is_conflict(edit_client) -> None:
    response = edit_client.post(
        "/api/v1/edit/images",
        files={"file": ("p.png", make_image_bytes("PNG", (800, 1000)), "image/png")},
    )
    image = response.json()["data"]
    analysis = edit_client.post(f"/api/v1/edit/images/{image['image_id']}/analyze")
    data = analysis.json()["data"]
    person = data["persons"][0]
    payload = {
        "image_id": image["image_id"],
        "person_id": person["person_id"],
        "region_type": "hat",
        "operation": "add",
        "style_id": "baseball_cap",
        "color_id": "black",
    }
    first = edit_client.post("/api/v1/edit/tasks", json=payload)
    second = edit_client.post("/api/v1/edit/tasks", json=payload)

    assert first.status_code == 201
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "EDIT_CONFLICT"
