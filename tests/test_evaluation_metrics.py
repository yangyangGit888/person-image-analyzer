"""评估指标单元测试。"""

from __future__ import annotations

from evaluation.metrics import evaluate, render_markdown, summarize_prediction


def _annotation(sample_id: str, *, pants="pants", socks="detected", shoes="detected", bare="not_detected"):
    return {
        "sample_id": sample_id,
        "annotations": [
            {
                "target": "lower_garment",
                "category": pants,
                "type": "shorts",
                "presence": "detected",
                "visibility": "clear",
            },
            {
                "target": "pantyhose",
                "category": "pantyhose",
                "type": "unknown",
                "presence": "not_detected",
                "visibility": "clear",
            },
            {
                "target": "ordinary_socks",
                "category": "ordinary_socks",
                "type": "crew",
                "presence": socks,
                "visibility": "clear",
            },
            {
                "target": "footwear",
                "category": "sneakers",
                "type": "sneakers",
                "presence": shoes,
                "visibility": "clear",
            },
            {
                "target": "bare_feet",
                "category": "bare",
                "type": "unknown",
                "presence": bare,
                "visibility": "clear",
            },
        ],
    }


def _analysis(
    *,
    pants_detected=True,
    socks_detected=True,
    shoes_detected=True,
    barefoot=False,
    side="unknown",
):
    items = []
    if pants_detected:
        items.append(
            {
                "category": "pants",
                "type": "shorts",
                "side": "not_applicable",
                "presence": "detected",
                "visibility": "clear",
            }
        )
    if socks_detected:
        items.append(
            {
                "category": "socks",
                "type": "crew",
                "side": side,
                "image_side": "left",
                "presence": "detected",
                "visibility": "clear",
            }
        )
    if shoes_detected:
        items.append(
            {
                "category": "shoes",
                "type": "sneakers",
                "side": side,
                "image_side": "left",
                "presence": "detected",
                "visibility": "clear",
            }
        )
    return {
        "items": items,
        "feet": [
            {
                "side": side,
                "image_side": "left",
                "barefoot": "barefoot" if barefoot else "not_barefoot",
                "visibility": "clear",
            }
        ],
    }


def test_summarize_prediction_aggregates_presence() -> None:
    summary = summarize_prediction(_analysis(side="unknown"))
    assert summary["lower_garment"]["category"] == "pants"
    assert summary["ordinary_socks"]["presence"] == "detected"
    assert summary["footwear"]["presence"] == "detected"
    assert summary["bare_feet"]["presence"] == "not_detected"


def test_evaluate_precision_recall_and_side_abstention() -> None:
    annotations = {
        "img_001": _annotation("img_001"),
        "img_002": _annotation("img_002", shoes="not_detected", bare="detected"),
    }
    predictions = {
        "img_001": {"sample_id": "img_001", "analysis": _analysis(side="left")},
        "img_002": {
            "sample_id": "img_002",
            "analysis": _analysis(shoes_detected=False, barefoot=True, side="unknown"),
        },
    }
    report = evaluate(annotations, predictions)

    assert report.samples == 2
    assert report.targets["footwear"].tp == 1
    assert report.targets["ordinary_socks"].tp == 2
    assert report.targets["bare_feet"].tp == 1
    assert report.side_unknown_items == 1
    assert report.side_items == 3


def test_uncertain_is_abstention_not_false_positive() -> None:
    annotations = {"img_001": _annotation("img_001", socks="detected")}
    analysis = _analysis(socks_detected=True)
    analysis["items"][1]["presence"] = "uncertain"
    analysis["items"][1]["visibility"] = "occluded"
    report = evaluate(annotations, {"img_001": {"analysis": analysis}})

    socks = report.targets["ordinary_socks"]
    assert socks.fp == 0
    assert socks.fn == 1
    assert socks.abstained == 1
    assert "识别质量评估报告" in render_markdown(report)
