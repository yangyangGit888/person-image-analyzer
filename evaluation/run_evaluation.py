"""运行真实图片评估集并生成指标报告。

用法：
    .venv/bin/python -m evaluation.run_evaluation --concurrency 2
    .venv/bin/python -m evaluation.run_evaluation --ids img_001,img_002
    .venv/bin/python -m evaluation.run_evaluation --force
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from app.ai.deepseek_provider import DeepSeekProvider
from app.ai.prompt_manager import PromptManager
from app.core.config import Settings
from app.core.exceptions import AppError
from app.services.analysis_service import AnalysisService
from app.services.image_analysis_service import ImageAnalysisService
from app.services.image_service import ImageService
from evaluation.metrics import evaluate, load_jsonl, render_markdown

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = ROOT / "evaluation" / "dataset" / "images"
DEFAULT_MANIFEST = ROOT / "evaluation" / "dataset" / "manifest.json"
DEFAULT_ANNOTATIONS = ROOT / "evaluation" / "annotations" / "ground_truth.jsonl"
DEFAULT_OUTPUT = ROOT / "evaluation" / "predictions" / "predictions.jsonl"
DEFAULT_REPORT = ROOT / "evaluation" / "reports" / "latest.md"
DEFAULT_JSON = ROOT / "evaluation" / "reports" / "latest.json"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Person Image Analyzer evaluation")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["sample_id"]: item for item in payload.get("images", [])}


def _load_done(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("sample_id"):
            result[str(record["sample_id"])] = record
    return result


async def _run_one(
    sample_id: str,
    image_path: Path,
    settings: Settings,
    image_service: ImageService,
    analysis_service: ImageAnalysisService,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        raw = image_path.read_bytes()
        processed = await image_service.process(raw)
        outcome = await analysis_service.analyze(
            processed,
            filename=image_path.name,
            request_id=sample_id,
        )
        return {
            "sample_id": sample_id,
            "analysis": outcome.analysis.model_dump(mode="json"),
            "meta": {
                "calls": outcome.calls,
                "usage": outcome.usage,
                "elapsed_seconds": round(time.perf_counter() - started, 2),
            },
        }
    except AppError as exc:
        return {
            "sample_id": sample_id,
            "error": exc.detail,
            "code": exc.code,
            "meta": {"elapsed_seconds": round(time.perf_counter() - started, 2)},
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "sample_id": sample_id,
            "error": f"{type(exc).__name__}: {exc}",
            "code": "EVALUATION_ERROR",
            "meta": {"elapsed_seconds": round(time.perf_counter() - started, 2)},
        }


async def _main() -> None:
    args = _args()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("app").setLevel(logging.WARNING)

    annotations = load_jsonl(str(args.annotations))
    manifest = _load_manifest(args.manifest)
    done = {} if args.force else _load_done(args.output)
    selected_ids = set(item.strip() for item in args.ids.split(",") if item.strip())
    sample_ids = [
        sample_id
        for sample_id in sorted(annotations)
        if not selected_ids or sample_id in selected_ids
    ]
    if args.limit:
        sample_ids = sample_ids[: args.limit]
    pending = [
        sample_id
        for sample_id in sample_ids
        if sample_id not in done or done[sample_id].get("error")
    ]

    settings = Settings()
    provider = DeepSeekProvider(settings)
    image_service = ImageService(settings)
    legacy = AnalysisService(provider, settings)
    analysis_service = ImageAnalysisService(
        provider,
        legacy,
        settings,
        prompt_manager=PromptManager(settings.prompts_path),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.parent.mkdir(parents=True, exist_ok=True)

    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    total_tokens = 0
    total_calls = 0

    print(
        f"[evaluation] samples={len(sample_ids)} pending={len(pending)} "
        f"concurrency={args.concurrency} model={settings.ai_model}",
        flush=True,
    )

    async def task(sample_id: str) -> None:
        nonlocal total_tokens, total_calls
        manifest_item = manifest.get(sample_id)
        if not manifest_item:
            record = {
                "sample_id": sample_id,
                "error": "manifest 中缺少该样本",
                "code": "MANIFEST_MISSING",
            }
        else:
            image_path = args.images / manifest_item["file"]
            async with semaphore:
                record = await _run_one(
                    sample_id,
                    image_path,
                    settings,
                    image_service,
                    analysis_service,
                )
        async with lock:
            with args.output.open("a", encoding="utf-8") as file_obj:
                file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
            meta = record.get("meta") or {}
            usage = meta.get("usage") or {}
            calls = int(meta.get("calls") or 0)
            total_calls += calls
            total_tokens += int(usage.get("total_tokens") or 0)
            status = "ERROR" if record.get("error") else "OK"
            print(
                f"[{len(done) + 1}/{len(sample_ids)}] {sample_id} {status} "
                f"calls={calls} total_tokens={usage.get('total_tokens', 0)} "
                f"elapsed={meta.get('elapsed_seconds', 0)}s",
                flush=True,
            )
            done[sample_id] = record

    try:
        await asyncio.gather(*(task(sample_id) for sample_id in pending))
    finally:
        await provider.aclose()

    predictions = _load_done(args.output)
    report = evaluate(annotations, predictions)
    args.report.write_text(render_markdown(report), encoding="utf-8")
    args.json_report.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[evaluation] run calls={total_calls} tokens={total_tokens} "
        f"errors={report.errors}",
        flush=True,
    )
    print(
        f"[evaluation] macro P/R/F1="
        f"{report.macro_precision}/{report.macro_recall}/{report.macro_f1}",
        flush=True,
    )
    print(f"[evaluation] report={args.report}", flush=True)
    print(f"[evaluation] json={args.json_report}", flush=True)


if __name__ == "__main__":
    asyncio.run(_main())
