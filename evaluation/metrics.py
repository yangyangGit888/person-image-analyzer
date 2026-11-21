"""图片识别评估指标。

原则：
- ``not_visible`` 是有效的可观察性判断，不等同于识别错误。
- ``uncertain`` 视为主动放弃判断，不强制计入误报。
- 当前标注集没有左右侧标签，因此 Side Accuracy 报告为 N/A；
  仍统计物品级 side unknown 比例。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

TARGETS = (
    "lower_garment",
    "pantyhose",
    "ordinary_socks",
    "footwear",
    "bare_feet",
)
SIDE_TARGETS = {"pantyhose", "ordinary_socks", "footwear"}

_VIS_RANK = {
    "not_visible": 0,
    "occluded": 1,
    "partial": 2,
    "clear": 3,
}


def normalize_visibility(value: Any) -> str:
    text = str(value or "not_visible").strip().lower()
    if text == "occluded":
        return "partial"
    if text in {"clear", "partial", "not_visible"}:
        return text
    return "not_visible"


def aggregate_presence(items: list[dict[str, Any]]) -> str:
    if not items:
        return "unknown"
    presences = [str(item.get("presence") or "uncertain") for item in items]
    visibilities = [str(item.get("visibility") or "not_visible") for item in items]
    if "detected" in presences:
        return "detected"
    if all(
        presence == "not_visible" or visibility == "not_visible"
        for presence, visibility in zip(presences, visibilities, strict=False)
    ):
        return "not_visible"
    if all(presence == "not_detected" for presence in presences):
        return "not_detected"
    return "uncertain"


def aggregate_visibility(items: list[dict[str, Any]]) -> str:
    if not items:
        return "not_visible"
    values = [normalize_visibility(item.get("visibility")) for item in items]
    return max(values, key=lambda value: _VIS_RANK.get(value, 0))


def _best_type(items: list[dict[str, Any]]) -> str:
    detected = [item for item in items if item.get("presence") == "detected"]
    pool = detected or items
    if not pool:
        return "unknown"
    counts = Counter(
        str(item.get("type") or "unknown")
        for item in pool
        if str(item.get("type") or "unknown") != "unknown"
    )
    if not counts:
        return "unknown"
    return counts.most_common(1)[0][0]


def summarize_prediction(analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = analysis.get("items") or []
    feet = analysis.get("feet") or []

    lower_items = [
        item
        for item in items
        if item.get("category") in {"pants", "skirt", "one_piece"}
    ]
    hosiery = [item for item in items if item.get("category") == "hosiery"]
    socks = [item for item in items if item.get("category") == "socks"]
    shoes = [item for item in items if item.get("category") == "shoes"]

    lower_detected = [
        item for item in lower_items if item.get("presence") == "detected"
    ]
    lower_category = "unknown"
    if lower_detected:
        lower_category = str(lower_detected[0].get("category") or "unknown")

    barefoot_values = [str(foot.get("barefoot") or "unknown") for foot in feet]
    if "barefoot" in barefoot_values:
        bare_presence = "detected"
    elif barefoot_values and all(value == "not_barefoot" for value in barefoot_values):
        bare_presence = "not_detected"
    elif barefoot_values and all(value == "not_visible" for value in barefoot_values):
        bare_presence = "not_visible"
    else:
        bare_presence = "unknown"
    bare_visibilities = [
        normalize_visibility(foot.get("visibility")) for foot in feet
    ]
    bare_visibility = (
        max(bare_visibilities, key=lambda value: _VIS_RANK.get(value, 0))
        if bare_visibilities
        else "not_visible"
    )

    return {
        "lower_garment": {
            "presence": aggregate_presence(lower_items),
            "visibility": aggregate_visibility(lower_items),
            "category": lower_category,
            "type": _best_type(lower_detected),
            "items": lower_items,
        },
        "pantyhose": {
            "presence": aggregate_presence(hosiery),
            "visibility": aggregate_visibility(hosiery),
            "category": (
                "pantyhose"
                if any(item.get("type") == "pantyhose" for item in hosiery)
                else ("hosiery" if hosiery else "unknown")
            ),
            "type": _best_type(hosiery),
            "items": hosiery,
        },
        "ordinary_socks": {
            "presence": aggregate_presence(socks),
            "visibility": aggregate_visibility(socks),
            "category": "ordinary_socks" if socks else "unknown",
            "type": _best_type(socks),
            "items": socks,
        },
        "footwear": {
            "presence": aggregate_presence(shoes),
            "visibility": aggregate_visibility(shoes),
            "category": _best_type(shoes),
            "type": _best_type(shoes),
            "items": shoes,
        },
        "bare_feet": {
            "presence": bare_presence,
            "visibility": bare_visibility,
            "category": "bare" if bare_presence == "detected" else "unknown",
            "type": "unknown",
            "items": feet,
        },
    }


@dataclass(slots=True)
class TargetScore:
    target: str
    total: int = 0
    gt_positive: int = 0
    predicted_positive: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    abstained: int = 0
    not_visible: int = 0
    visibility_total: int = 0
    visibility_correct: int = 0

    @property
    def precision(self) -> float | None:
        return self.tp / self.predicted_positive if self.predicted_positive else None

    @property
    def recall(self) -> float | None:
        return self.tp / self.gt_positive if self.gt_positive else None

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    @property
    def abstention_rate(self) -> float:
        return self.abstained / self.total if self.total else 0.0

    @property
    def visibility_accuracy(self) -> float | None:
        if not self.visibility_total:
            return None
        return self.visibility_correct / self.visibility_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "total": self.total,
            "gt_positive": self.gt_positive,
            "predicted_positive": self.predicted_positive,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "abstention_rate": self.abstention_rate,
            "not_visible_rate": (
                self.not_visible / self.total if self.total else 0.0
            ),
            "visibility_accuracy": self.visibility_accuracy,
        }


@dataclass(slots=True)
class EvaluationReport:
    samples: int = 0
    errors: int = 0
    targets: dict[str, TargetScore] = field(default_factory=dict)
    lower_category_total: int = 0
    lower_category_correct: int = 0
    socks_type_total: int = 0
    socks_type_correct: int = 0
    footwear_type_total: int = 0
    footwear_type_correct: int = 0
    footwear_group_total: int = 0
    footwear_group_correct: int = 0
    side_items: int = 0
    side_unknown_items: int = 0
    sample_rows: list[dict[str, Any]] = field(default_factory=list)
    error_rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def macro_precision(self) -> float | None:
        values = [
            score.precision for score in self.targets.values() if score.precision is not None
        ]
        return sum(values) / len(values) if values else None

    @property
    def macro_recall(self) -> float | None:
        values = [
            score.recall for score in self.targets.values() if score.recall is not None
        ]
        return sum(values) / len(values) if values else None

    @property
    def macro_f1(self) -> float | None:
        values = [score.f1 for score in self.targets.values() if score.f1 is not None]
        return sum(values) / len(values) if values else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "errors": self.errors,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "targets": {
                name: score.to_dict() for name, score in self.targets.items()
            },
            "category_accuracy": {
                "lower_garment": _ratio(
                    self.lower_category_correct,
                    self.lower_category_total,
                ),
                "ordinary_socks_type": _ratio(
                    self.socks_type_correct,
                    self.socks_type_total,
                ),
                "footwear_type": _ratio(
                    self.footwear_type_correct,
                    self.footwear_type_total,
                ),
                "footwear_group": _ratio(
                    self.footwear_group_correct,
                    self.footwear_group_total,
                ),
            },
            "side_accuracy": None,
            "side_abstention_rate": _ratio(self.side_unknown_items, self.side_items),
            "samples_detail": self.sample_rows,
            "error_rows": self.error_rows,
        }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _gt_annotation(annotations: list[dict[str, Any]], target: str) -> dict[str, Any]:
    for annotation in annotations:
        if annotation.get("target") == target:
            return annotation
    return {}


def _footwear_group(value: str) -> str:
    token = str(value or "unknown").lower()
    if token in {"sneakers", "heels", "flats", "loafers", "boots", "sandals", "slippers"}:
        return token
    if token in {"clogs", "mary_janes"}:
        return "other"
    return "unknown"


def evaluate(
    annotations: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> EvaluationReport:
    report = EvaluationReport(
        samples=len(annotations),
        targets={target: TargetScore(target=target) for target in TARGETS},
    )
    for sample_id, annotation in sorted(annotations.items()):
        prediction = predictions.get(sample_id)
        if not prediction or prediction.get("error"):
            report.errors += 1
            report.error_rows.append(
                {
                    "sample_id": sample_id,
                    "error": (prediction or {}).get("error", "missing prediction"),
                }
            )
            continue
        analysis = prediction.get("analysis") or prediction
        summary = summarize_prediction(analysis)
        annotations_list = annotation.get("annotations") or []
        sample_row: dict[str, Any] = {"sample_id": sample_id, "targets": {}}

        for target in TARGETS:
            gt = _gt_annotation(annotations_list, target)
            if not gt:
                continue
            gt_presence = str(gt.get("presence") or "unknown")
            gt_visibility = normalize_visibility(gt.get("visibility"))
            predicted = summary[target]
            pred_presence = str(predicted.get("presence") or "unknown")
            pred_visibility = normalize_visibility(predicted.get("visibility"))
            score = report.targets[target]
            score.total += 1
            if gt_presence == "detected":
                score.gt_positive += 1
            if pred_presence == "detected":
                score.predicted_positive += 1

            if gt_presence == "detected" and pred_presence == "detected":
                score.tp += 1
            elif gt_presence != "detected" and pred_presence == "detected":
                score.fp += 1
            elif gt_presence == "detected" and pred_presence != "detected":
                score.fn += 1

            if pred_presence == "uncertain":
                score.abstained += 1
            if pred_presence == "not_visible":
                score.not_visible += 1
            score.visibility_total += 1
            if pred_visibility == gt_visibility:
                score.visibility_correct += 1

            if target == "lower_garment":
                gt_category = str(gt.get("category") or "unknown")
                pred_category = str(predicted.get("category") or "unknown")
                if gt_category != "unknown":
                    report.lower_category_total += 1
                    if gt_category == pred_category:
                        report.lower_category_correct += 1
            elif target == "ordinary_socks":
                gt_type = str(gt.get("type") or "unknown")
                pred_type = str(predicted.get("type") or "unknown")
                if gt_type != "unknown" and gt_presence == "detected":
                    report.socks_type_total += 1
                    if gt_type == pred_type:
                        report.socks_type_correct += 1
            elif target == "footwear":
                gt_type = str(gt.get("type") or "unknown")
                pred_type = str(predicted.get("type") or "unknown")
                if gt_type != "unknown" and gt_presence == "detected":
                    report.footwear_type_total += 1
                    if gt_type == pred_type:
                        report.footwear_type_correct += 1
                    report.footwear_group_total += 1
                    if _footwear_group(gt_type) == _footwear_group(pred_type):
                        report.footwear_group_correct += 1

            if target in SIDE_TARGETS:
                for item in predicted.get("items", []):
                    if item.get("presence") != "detected":
                        continue
                    report.side_items += 1
                    if str(item.get("side") or "unknown") == "unknown":
                        report.side_unknown_items += 1

            sample_row["targets"][target] = {
                "gt_presence": gt_presence,
                "pred_presence": pred_presence,
                "gt_visibility": gt_visibility,
                "pred_visibility": pred_visibility,
                "gt_category": gt.get("category"),
                "pred_category": predicted.get("category"),
                "gt_type": gt.get("type"),
                "pred_type": predicted.get("type"),
                "gt_side": gt.get("side"),
                "pred_sides": [
                    item.get("side")
                    for item in predicted.get("items", [])
                    if item.get("presence") == "detected"
                ],
            }
        report.sample_rows.append(sample_row)
    return report


def load_jsonl(path: str) -> dict[str, dict[str, Any]]:
    import json
    from pathlib import Path

    result: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = record.get("sample_id")
        if sample_id:
            result[str(sample_id)] = record
    return result


def render_markdown(report: EvaluationReport) -> str:
    lines = [
        "# 识别质量评估报告",
        "",
        f"- 样本数：{report.samples}",
        f"- 失败样本：{report.errors}",
        f"- Macro Precision：{_fmt(report.macro_precision)}",
        f"- Macro Recall：{_fmt(report.macro_recall)}",
        f"- Macro F1：{_fmt(report.macro_f1)}",
        "",
        "## 分目标指标",
        "",
        "| 目标 | N | GT 正例 | TP | FP | FN | Precision | Recall | F1 | Visibility Acc | Abstention | Not Visible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        score = report.targets[target]
        lines.append(
            "| "
            + " | ".join(
                [
                    target,
                    str(score.total),
                    str(score.gt_positive),
                    str(score.tp),
                    str(score.fp),
                    str(score.fn),
                    _fmt(score.precision),
                    _fmt(score.recall),
                    _fmt(score.f1),
                    _fmt(score.visibility_accuracy),
                    f"{score.abstention_rate:.1%}",
                    f"{(score.not_visible / score.total if score.total else 0):.1%}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 类别与侧向",
            "",
            f"- 下装 Category Accuracy：{_fmt(_ratio(report.lower_category_correct, report.lower_category_total))} ({report.lower_category_correct}/{report.lower_category_total})",
            f"- 普通袜子 Type Accuracy：{_fmt(_ratio(report.socks_type_correct, report.socks_type_total))} ({report.socks_type_correct}/{report.socks_type_total})",
            f"- 鞋子 Type Accuracy：{_fmt(_ratio(report.footwear_type_correct, report.footwear_type_total))} ({report.footwear_type_correct}/{report.footwear_type_total})",
            f"- 鞋子 Group Accuracy：{_fmt(_ratio(report.footwear_group_correct, report.footwear_group_total))} ({report.footwear_group_correct}/{report.footwear_group_total})",
            "- Side Accuracy：N/A（当前标注集没有左右侧标签）",
            f"- Side Abstention Rate：{_fmt(_ratio(report.side_unknown_items, report.side_items))} ({report.side_unknown_items}/{report.side_items})",
            "",
            "## 样本明细",
            "",
            "| 样本 | lower | pantyhose | socks | footwear | bare feet |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in report.sample_rows:
        cells = [row["sample_id"]]
        for target in TARGETS:
            item = row["targets"].get(target)
            if not item:
                cells.append("-")
            else:
                cells.append(
                    f"{item['gt_presence']}→{item['pred_presence']} / "
                    f"{item['gt_visibility']}→{item['pred_visibility']}"
                )
        lines.append("| " + " | ".join(cells) + " |")
    if report.error_rows:
        lines.extend(["", "## 失败样本", ""])
        for row in report.error_rows:
            lines.append(f"- `{row['sample_id']}`: {row['error']}")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"
