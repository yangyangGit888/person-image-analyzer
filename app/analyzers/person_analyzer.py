"""全图人物与下装概览分析。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from app.ai.base import BaseVisionProvider, ImageInput, ModelResult
from app.ai.prompt_manager import PromptManager
from app.analyzers.models import OverviewPayload
from app.analyzers.parsing import geometry_from_box, warnings
from app.core.exceptions import AIInvalidResponseError, NoPersonDetectedError


class LegacyPayloadDetected(Exception):
    """旧版单次调用响应，交给兼容回退路径处理。"""

    def __init__(self, result: ModelResult, prompt_versions: dict[str, str]) -> None:
        self.result = result
        self.prompt_versions = prompt_versions
        super().__init__("旧版模型响应")


@dataclass(slots=True)
class OverviewObservation:
    payload: OverviewPayload
    result: ModelResult
    prompt_versions: dict[str, str]
    person_bbox: list[int] | None = None
    region_boxes: dict[str, list[int] | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _bbox(
    value: Any,
    *,
    source_width: int,
    source_height: int,
) -> list[int] | None:
    geometry = geometry_from_box(
        value,
        source_width=source_width,
        source_height=source_height,
        method="overview_estimate",
    )
    return geometry.bbox if geometry is not None else None


async def analyze_overview(
    provider: BaseVisionProvider,
    image: ImageInput,
    prompt_manager: PromptManager,
    *,
    source_width: int,
    source_height: int,
) -> OverviewObservation:
    """调用全图 overview Prompt，并解析人物/区域位置。"""

    system = prompt_manager.load("image_overview_system")
    user = prompt_manager.load("image_overview_user")
    prompt = user.render(image_width=image.width, image_height=image.height)
    result = await provider.analyze_image(
        image,
        prompt,
        system_prompt=system.content,
    )

    if isinstance(result.parsed.get("persons"), list):
        raise LegacyPayloadDetected(
            result,
            {
                "image_overview_system": system.version,
                "image_overview_user": user.version,
            },
        )
    if not any(
        key in result.parsed
        for key in ("lower_body", "body_observability", "person_orientation")
    ):
        raise AIInvalidResponseError(
            detail="概览结果缺少 lower_body、body_observability 或 person_orientation",
        )

    try:
        payload = OverviewPayload.model_validate(result.parsed)
    except ValidationError as exc:
        raise AIInvalidResponseError(
            detail="概览结果缺少 person_count 或字段类型不正确",
        ) from exc

    if payload.person_count <= 0:
        raise NoPersonDetectedError()

    person_bbox = _bbox(
        payload.person_bbox,
        source_width=source_width,
        source_height=source_height,
    )
    region_payloads = payload.body_observability
    region_boxes = {
        "lower_body": _bbox(
            region_payloads.lower_body.bbox or payload.lower_body.bbox,
            source_width=source_width,
            source_height=source_height,
        ),
        "left_leg": _bbox(
            region_payloads.left_leg.bbox,
            source_width=source_width,
            source_height=source_height,
        ),
        "right_leg": _bbox(
            region_payloads.right_leg.bbox,
            source_width=source_width,
            source_height=source_height,
        ),
        "left_foot": _bbox(
            region_payloads.left_foot.bbox,
            source_width=source_width,
            source_height=source_height,
        ),
        "right_foot": _bbox(
            region_payloads.right_foot.bbox,
            source_width=source_width,
            source_height=source_height,
        ),
    }

    return OverviewObservation(
        payload=payload,
        result=result,
        prompt_versions={
            "image_overview_system": system.version,
            "image_overview_user": user.version,
        },
        person_bbox=person_bbox,
        region_boxes=region_boxes,
        warnings=warnings(payload.warnings),
    )
