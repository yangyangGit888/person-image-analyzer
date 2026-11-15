"""测试辅助工具。"""

from __future__ import annotations

import copy
import json
from io import BytesIO
from typing import Any

from PIL import Image
from pillow_heif import register_heif_opener

from app.ai.base import BaseVisionProvider, ImageInput, ModelResult

register_heif_opener()


def make_image_bytes(
    fmt: str = "PNG",
    size: tuple[int, int] = (640, 480),
    color: tuple[int, int, int] = (180, 120, 80),
) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()


def single_person_result() -> dict[str, Any]:
    return {
        "person_count": 1,
        "persons": [
            {
                "person_id": 1,
                "position": {
                    "description": "图片中央",
                    "bbox": {"x": 0.3, "y": 0.1, "width": 0.35, "height": 0.85},
                },
                "visibility": {
                    "full_body_visible": "unknown",
                    "head_visible": "true",
                    "upper_body_visible": "true",
                    "legs_visible": "true",
                    "feet_visible": "unknown",
                },
                "head": {
                    "wearing_hat": "true",
                    "hat_type": "棒球帽",
                    "hat_color": "黑色",
                    "wearing_glasses": "false",
                },
                "upper_body": {
                    "wearing_top": "true",
                    "top_type": "T恤",
                    "top_color": "白色",
                    "wearing_jacket": "false",
                    "jacket_color": "unknown",
                },
                "lower_body": {
                    "wearing_pants": "true",
                    "pants_type": "牛仔裤",
                    "pants_color": "蓝色",
                    "wearing_skirt": "false",
                    "skirt_color": "unknown",
                },
                "legs": {
                    "skin_visible": "true",
                    "wearing_tights": "unknown",
                    "wearing_stockings": "unknown",
                    "wearing_long_pants": "true",
                    "wearing_shorts": "false",
                    "leg_covering": "牛仔裤",
                },
                "feet": {
                    "wearing_socks": "unknown",
                    "socks_type": "unknown",
                    "socks_color": "unknown",
                    "wearing_shoes": "unknown",
                    "shoes_type": "unknown",
                    "shoes_color": "unknown",
                    "barefoot": "unknown",
                },
                "confidence": {
                    "head": 0.95,
                    "upper_body": 0.92,
                    "lower_body": 0.9,
                    "legs": 0.75,
                    "feet": 0.3,
                },
                "evidence": {"feet": "脚部被画面裁切，无法判断袜子与鞋子"},
            }
        ],
    }


def two_person_result() -> dict[str, Any]:
    return {
        "person_count": 2,
        "persons": [
            {
                "person_id": 1,
                "position": {"description": "图片左侧", "bbox": None},
                "head": {
                    "wearing_hat": True,
                    "hat_type": "毛线帽",
                    "hat_color": "灰色",
                    "wearing_glasses": False,
                },
                "upper_body": {
                    "wearing_top": True,
                    "top_type": "卫衣",
                    "top_color": "红色",
                    "wearing_jacket": True,
                    "jacket_color": "黑色",
                },
                "lower_body": {
                    "wearing_pants": True,
                    "pants_type": "运动裤",
                    "pants_color": "黑色",
                    "wearing_skirt": False,
                    "skirt_color": "unknown",
                },
                "legs": {"skin_visible": False, "wearing_long_pants": True},
                "feet": {
                    "wearing_socks": True,
                    "socks_type": "短袜",
                    "socks_color": "白色",
                    "wearing_shoes": True,
                    "shoes_type": "运动鞋",
                    "shoes_color": "白色",
                    "barefoot": False,
                },
                "confidence": {
                    "head": 0.9,
                    "upper_body": 0.88,
                    "lower_body": 0.86,
                    "legs": 0.8,
                    "feet": 0.7,
                },
                "evidence": {},
            },
            {
                "person_id": 2,
                "position": {"description": "图片右侧", "bbox": None},
                "head": {
                    "wearing_hat": False,
                    "hat_type": "unknown",
                    "hat_color": "unknown",
                    "wearing_glasses": True,
                },
                "upper_body": {
                    "wearing_top": True,
                    "top_type": "衬衫",
                    "top_color": "蓝色",
                    "wearing_jacket": False,
                    "jacket_color": "unknown",
                },
                "lower_body": {
                    "wearing_pants": False,
                    "pants_type": "unknown",
                    "pants_color": "unknown",
                    "wearing_skirt": True,
                    "skirt_color": "白色",
                },
                "legs": {
                    "skin_visible": True,
                    "wearing_tights": True,
                    "wearing_stockings": "unknown",
                },
                "feet": {
                    "wearing_socks": "unknown",
                    "wearing_shoes": "unknown",
                    "barefoot": "unknown",
                },
                "confidence": {"head": 0.8, "feet": None},
                "evidence": {"feet": "脚部在画面外"},
            },
        ],
    }


class FakeProvider(BaseVisionProvider):
    """用于 API 测试的假 Provider，不发起任何网络请求。"""

    provider_name = "deepseek"
    model = "deepseek-flash"
    base_url = "https://api.deepseek.com"
    supports_vision = True

    def __init__(
        self,
        parsed: dict[str, Any] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.parsed = copy.deepcopy(parsed) if parsed is not None else single_person_result()
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def analyze_image(
        self,
        image: ImageInput,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> ModelResult:
        self.calls.append(
            {
                "image": image,
                "prompt": prompt,
                "system_prompt": system_prompt,
            }
        )
        if self.error is not None:
            raise self.error
        return ModelResult(
            content=json.dumps(self.parsed, ensure_ascii=False),
            parsed=copy.deepcopy(self.parsed),
            model=self.model,
            provider=self.provider_name,
            usage={"prompt_tokens": 1200, "completion_tokens": 400},
            request_id="fake-request-id",
        )

    async def aclose(self) -> None:
        return None
