"""图片分析 API 测试客户端。

用法：
    python scripts/test_client.py ./photo.jpg
    python scripts/test_client.py ./photo.jpg --url http://127.0.0.1:8000/api/v1/analyze
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from pathlib import Path

import httpx

DEFAULT_URL = "http://127.0.0.1:8000/api/v1/analyze"


def main() -> int:
    parser = argparse.ArgumentParser(description="上传本地图片并打印人物识别结果")
    parser.add_argument("image", type=Path, help="本地图片路径")
    parser.add_argument("--url", default=DEFAULT_URL, help="分析接口地址")
    parser.add_argument("--timeout", type=float, default=300.0, help="请求超时秒数")
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"文件不存在: {args.image}", file=sys.stderr)
        return 1

    content_type = mimetypes.guess_type(args.image.name)[0] or "application/octet-stream"
    try:
        with args.image.open("rb") as file_obj:
            response = httpx.post(
                args.url,
                files={"file": (args.image.name, file_obj, content_type)},
                timeout=args.timeout,
            )
    except httpx.RequestError as exc:
        print(f"请求失败: {exc}", file=sys.stderr)
        return 1

    print(f"HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError:
        print(response.text)
        return 1

    print(json.dumps(body, ensure_ascii=False, indent=2))
    return 0 if response.is_success and body.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
