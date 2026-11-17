"""人像局部编辑 API 测试客户端。

用法：
    python scripts/test_edit_client.py ./portrait.jpg
    python scripts/test_edit_client.py ./portrait.jpg --region hat --style beret --color red
    python scripts/test_edit_client.py ./portrait.jpg --operation add --person 1 --output result.png
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="上传图片、分析人物并执行局部编辑")
    parser.add_argument("image", type=Path, help="本地图片路径")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--person", type=int, default=1, help="人物序号，从 1 开始")
    parser.add_argument("--region", default="hat")
    parser.add_argument("--operation", choices=["add", "replace", "auto"], default="auto")
    parser.add_argument("--style", default="baseball_cap")
    parser.add_argument("--color", default="black")
    parser.add_argument("--output", type=Path, default=Path("edit-result.png"))
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"文件不存在: {args.image}", file=sys.stderr)
        return 1

    base_url = args.base_url.rstrip("/")
    content_type = mimetypes.guess_type(args.image.name)[0] or "application/octet-stream"
    try:
        with httpx.Client(timeout=args.timeout) as client:
            with args.image.open("rb") as file_obj:
                upload = client.post(
                    f"{base_url}/api/v1/edit/images",
                    files={"file": (args.image.name, file_obj, content_type)},
                )
            upload.raise_for_status()
            image = upload.json()["data"]
            print(f"上传成功: image_id={image['image_id']} {image['width']}x{image['height']}")

            analyzed = client.post(
                f"{base_url}/api/v1/edit/images/{image['image_id']}/analyze"
            )
            analyzed.raise_for_status()
            data = analyzed.json()["data"]
            person = next(
                (item for item in data["persons"] if item["ordinal"] == args.person),
                None,
            )
            if person is None:
                print(f"未找到人物序号 {args.person}", file=sys.stderr)
                return 1
            region = next(
                (item for item in data["regions"] if item["region_type"] == args.region),
                None,
            )
            if region is None:
                print(f"未找到区域 {args.region}", file=sys.stderr)
                return 1

            print(
                f"人物 {person['ordinal']} bbox={person['bbox']} mask={person['mask_ready']}"
            )
            print(
                f"区域 {region['region_type']} existence={region['existence_status']} "
                f"mask={region['mask_ready']} strategy={region['mask_strategy']}"
            )
            if not region["mask_ready"]:
                print("该区域没有可用 Mask，可能被遮挡或分割质量不足", file=sys.stderr)
                return 1

            operation = args.operation
            if operation == "auto":
                operation = "replace" if region["existence_status"] == "existing" else "add"
            if operation == "replace" and region["mask_strategy"] != "object":
                print("该区域没有可替换的物体 Mask，请使用 --operation add", file=sys.stderr)
                return 1

            task_response = client.post(
                f"{base_url}/api/v1/edit/tasks",
                json={
                    "image_id": image["image_id"],
                    "person_id": person["person_id"],
                    "region_type": args.region,
                    "operation": operation,
                    "style_id": args.style,
                    "color_id": args.color,
                },
            )
            if task_response.status_code >= 400:
                print(task_response.text, file=sys.stderr)
                return 1
            task = task_response.json()["data"]
            print(f"任务已创建: {task['task_id']} ({operation})")
            print(task["prompt"])

            generation_response = client.post(
                f"{base_url}/api/v1/edit/generations",
                json={
                    "image_id": image["image_id"],
                    "person_id": person["person_id"],
                    "task_ids": [task["task_id"]],
                },
            )
            generation_response.raise_for_status()
            generation_id = generation_response.json()["data"]["generation"]["generation_id"]
            print(f"生成任务已提交: {generation_id}")

            deadline = time.time() + args.timeout
            while time.time() < deadline:
                detail = client.get(
                    f"{base_url}/api/v1/edit/generations/{generation_id}"
                ).json()["data"]
                status = detail["generation"]["status"]
                print(f"状态: {status}")
                if status == "succeeded":
                    result = client.get(
                        f"{base_url}/api/v1/edit/generations/{generation_id}/result"
                    )
                    result.raise_for_status()
                    args.output.write_bytes(result.content)
                    print(f"结果已保存: {args.output.resolve()}")
                    return 0
                if status == "failed":
                    print(
                        f"生成失败: {detail['generation'].get('error_code')} "
                        f"{detail['generation'].get('error_detail')}",
                        file=sys.stderr,
                    )
                    return 1
                time.sleep(3)
    except httpx.HTTPError as exc:
        print(f"请求失败: {exc}", file=sys.stderr)
        return 1

    print("等待生成超时", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
