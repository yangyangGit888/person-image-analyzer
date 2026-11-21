"""V2 图片分析测试用 Mock Provider。"""

from __future__ import annotations

import copy
import json
from typing import Any

from app.ai.base import BaseVisionProvider, ImageInput, ModelResult
from tests.helpers import two_person_result


def _region(visibility: str, bbox: dict[str, float] | None = None, *reasons: str):
    return {
        "visibility": visibility,
        "occlusion_reason": list(reasons),
        "bbox": bbox,
        "confidence": "high",
    }


def _overview(
    *,
    category: str,
    item_type: str,
    attributes: dict[str, Any],
    lower_visibility: str = "clear",
    legs_visibility: str = "clear",
    feet_visibility: str = "clear",
    reasons: list[str] | None = None,
    person_count: int = 1,
) -> dict[str, Any]:
    reasons = reasons or []
    return {
        "person_count": person_count,
        "person_orientation": "front",
        "person_pose": "standing",
        "person_bbox": {"x": 0.25, "y": 0.05, "width": 0.5, "height": 0.9},
        "lower_body": {
            "category": category,
            "type": item_type,
            "presence": "detected",
            "visibility": lower_visibility,
            "style_tags": ["casual"],
            "design_features": ["solid_color"],
            "attributes": attributes,
            "bbox": {"x": 0.3, "y": 0.45, "width": 0.4, "height": 0.45},
            "occlusion_reason": reasons,
            "confidence": "high",
        },
        "body_observability": {
            "lower_body": _region(
                lower_visibility,
                {"x": 0.3, "y": 0.45, "width": 0.4, "height": 0.45},
                *reasons,
            ),
            "left_leg": _region(legs_visibility, None, *reasons),
            "right_leg": _region(legs_visibility, None, *reasons),
            "left_foot": _region(feet_visibility, None, *reasons),
            "right_foot": _region(feet_visibility, None, *reasons),
        },
        "warnings": [],
    }


def _clothing(
    category: str,
    item_type: str,
    attributes: dict[str, Any],
) -> dict[str, Any]:
    return {
        "category": category,
        "type": item_type,
        "presence": "detected",
        "visibility": "clear",
        "style_tags": ["casual"],
        "design_features": ["solid_color"],
        "attributes": attributes,
        "bbox": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8},
        "occlusion_reason": [],
        "confidence": "high",
        "evidence": ["局部裁剪可见下装"],
        "warnings": [],
    }


def _legs(
    item_type: str = "unknown",
    presence: str = "not_detected",
    *,
    pair_connection: bool | str = "unknown",
    color: str = "unknown",
    side_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    side = {
        "presence": presence,
        "visibility": "clear",
        "type": item_type,
        "attributes": {
            "color": color,
            "visual_material": "sheer" if item_type == "pantyhose" else "unknown",
            "texture": "smooth" if item_type == "pantyhose" else "unknown",
            "visual_thickness": "thin" if item_type == "pantyhose" else "unknown",
            "sheerness_level": 70 if item_type == "pantyhose" else None,
            "covers_feet": True if item_type == "pantyhose" else None,
        },
        "bbox": {"x": 0.05, "y": 0.1, "width": 0.4, "height": 0.85}
        if presence == "detected"
        else None,
        "occlusion_reason": [],
        "confidence": "high" if presence == "detected" else "medium",
        "evidence": ["腿部裁剪观察"],
    }
    if side_fields:
        side.update(side_fields)
    return {
        "left_leg": copy.deepcopy(side),
        "right_leg": copy.deepcopy(side),
        "pair_connection_visible": pair_connection,
        "warnings": [],
    }


def _side_foot(
    *,
    visibility: str = "clear",
    barefoot: str = "not_barefoot",
    socks_type: str = "ankle",
    shoes_type: str = "sneakers",
    socks_presence: str = "detected",
    shoes_presence: str = "detected",
    skin_visible: bool = False,
    side_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sock = {
        "presence": socks_presence,
        "visibility": "clear",
        "type": socks_type,
        "attributes": {"color": "white", "covers_feet": True},
        "bbox": {"x": 0.1, "y": 0.35, "width": 0.3, "height": 0.5}
        if socks_presence == "detected"
        else None,
        "occlusion_reason": [],
        "confidence": "high",
        "evidence": ["脚部裁剪观察"],
    }
    shoe = {
        "presence": shoes_presence,
        "visibility": "clear",
        "type": shoes_type,
        "attributes": {"color": "white", "shoe_shape": "low_top"},
        "bbox": {"x": 0.1, "y": 0.6, "width": 0.35, "height": 0.35}
        if shoes_presence == "detected"
        else None,
        "occlusion_reason": [],
        "confidence": "high",
        "evidence": ["脚部裁剪观察"],
    }
    result = {
        "visibility": visibility,
        "barefoot": barefoot,
        "skin_visible": skin_visible,
        "socks": sock,
        "shoes": shoe,
        "occlusion_reason": [],
        "confidence": "high",
        "evidence": ["脚部裁剪观察"],
    }
    if side_fields:
        result.update(side_fields)
    return result


def _feet(
    *,
    left: dict[str, Any] | None = None,
    right: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "left_foot": left or _side_foot(),
        "right_foot": right or _side_foot(),
        "warnings": [],
    }


SCENARIOS: dict[str, dict[str, Any]] = {
    "normal": {
        "overview": _overview(
            category="pants",
            item_type="jeans",
            attributes={"color": "blue", "length": "long", "pattern": "solid"},
        ),
        "clothing": _clothing(
            "pants",
            "jeans",
            {"color": "blue", "length": "long", "pattern": "solid"},
        ),
        "legs": _legs(),
        "feet": _feet(),
    },
    "long_skirt": {
        "overview": _overview(
            category="one_piece",
            item_type="maxi_dress",
            attributes={"color": "black", "length": "long", "pattern": "solid"},
            legs_visibility="not_visible",
            feet_visibility="not_visible",
            reasons=["long_skirt"],
        ),
        "clothing": _clothing(
            "one_piece",
            "maxi_dress",
            {"color": "black", "length": "long", "pattern": "solid"},
        ),
        "legs": None,
        "feet": None,
    },
    "hosiery_socks_shoes": {
        "overview": _overview(
            category="skirt",
            item_type="pleated_skirt",
            attributes={"color": "navy", "length": "short", "pattern": "solid"},
        ),
        "clothing": _clothing(
            "skirt",
            "pleated_skirt",
            {"color": "navy", "length": "short", "pattern": "solid"},
        ),
        "legs": _legs(
            "pantyhose",
            "detected",
            pair_connection=True,
            color="black",
        ),
        "feet": _feet(),
    },
    "one_shoe": {
        "overview": _overview(
            category="pants",
            item_type="casual_pants",
            attributes={"color": "black", "length": "long", "pattern": "solid"},
        ),
        "clothing": _clothing(
            "pants",
            "casual_pants",
            {"color": "black", "length": "long", "pattern": "solid"},
        ),
        "legs": _legs(),
        "feet": _feet(
            left=_side_foot(),
            right=_side_foot(
                barefoot="barefoot",
                socks_type="unknown",
                shoes_type="unknown",
                socks_presence="not_detected",
                shoes_presence="not_detected",
                skin_visible=True,
            ),
        ),
    },
    "multi": {
        "overview": _overview(
            category="pants",
            item_type="jeans",
            attributes={"color": "blue", "length": "long", "pattern": "solid"},
            person_count=2,
        ),
        "clothing": _clothing(
            "pants",
            "jeans",
            {"color": "blue", "length": "long", "pattern": "solid"},
        ),
        "legs": _legs(),
        "feet": _feet(),
    },
    "side_unknown": {
        "overview": _overview(
            category="pants",
            item_type="casual_pants",
            attributes={"color": "black", "length": "long", "pattern": "solid"},
        ),
        "clothing": _clothing(
            "pants",
            "casual_pants",
            {"color": "black", "length": "long", "pattern": "solid"},
        ),
        "legs": _legs(
            "thigh_high",
            "detected",
            color="black",
            side_fields={
                "person_side": "unknown",
                "side_basis": "unknown",
                "side_confidence": "low",
            },
        ),
        "feet": _feet(
            left=_side_foot(
                side_fields={
                    "person_side": "unknown",
                    "side_basis": "unknown",
                    "side_confidence": "low",
                }
            ),
            right=_side_foot(
                side_fields={
                    "person_side": "unknown",
                    "side_basis": "unknown",
                    "side_confidence": "low",
                }
            ),
        ),
    },
    "partial_feet": {
        "overview": _overview(
            category="skirt",
            item_type="a_line_skirt",
            attributes={"color": "white", "length": "short", "pattern": "solid"},
            feet_visibility="not_visible",
            reasons=["out_of_frame"],
        ),
        "clothing": _clothing(
            "skirt",
            "a_line_skirt",
            {"color": "white", "length": "short", "pattern": "solid"},
        ),
        "legs": _legs("thigh_high", "detected", color="black"),
        "feet": None,
    },
}


class V2FakeProvider(BaseVisionProvider):
    """按 Prompt 内容返回 overview / clothing / legs / feet 结果。"""

    provider_name = "deepseek"
    model = "deepseek-flash"
    base_url = "https://api.deepseek.com"
    supports_vision = True

    def __init__(self, scenario: str = "normal") -> None:
        self.scenario = scenario
        self.calls: list[dict[str, Any]] = []

    def _response(self, prompt: str) -> dict[str, Any]:
        if "所有真实人物" in prompt:
            return two_person_result()
        if "下装区域" in prompt:
            return SCENARIOS[self.scenario]["clothing"]
        if "腿部区域" in prompt:
            return SCENARIOS[self.scenario]["legs"]
        if "脚部区域" in prompt:
            return SCENARIOS[self.scenario]["feet"]
        return SCENARIOS[self.scenario]["overview"]

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
        parsed = copy.deepcopy(self._response(prompt))
        return ModelResult(
            content=json.dumps(parsed, ensure_ascii=False),
            parsed=parsed,
            model=self.model,
            provider=self.provider_name,
            usage={"prompt_tokens": 800, "completion_tokens": 300},
            request_id="v2-fake-request",
        )

    async def aclose(self) -> None:
        return None
