"""左右脚普通袜子、鞋子和赤脚状态的局部调用。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from app.ai.base import BaseVisionProvider, ModelResult
from app.ai.prompt_manager import PromptManager
from app.analyzers.models import FeetDetailPayload, FeetSidePayload
from app.analyzers.parsing import geometry_from_box, warnings
from app.core.exceptions import AIInvalidResponseError
from app.processing.image_processor import CropRegion
from app.schemas.analysis_v2 import Geometry, Side


@dataclass(slots=True)
class FeetObservation:
    payload: FeetDetailPayload
    result: ModelResult
    prompt_versions: dict[str, str]
    sock_geometries: dict[Side, Geometry | None] = field(default_factory=dict)
    shoe_geometries: dict[Side, Geometry | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def side_payload(self, side: Side) -> FeetSidePayload:
        return self.payload.left_foot if side is Side.LEFT else self.payload.right_foot


def _geometry(
    payload: Any,
    crop: CropRegion,
    source_width: int,
    source_height: int,
) -> Geometry | None:
    if payload is None:
        return None
    return geometry_from_box(
        payload.bbox,
        source_width=source_width,
        source_height=source_height,
        crop=crop,
        method="crop_estimate",
    )


async def analyze_feet(
    provider: BaseVisionProvider,
    crop: CropRegion,
    prompt_manager: PromptManager,
    *,
    source_width: int,
    source_height: int,
) -> FeetObservation:
    system = prompt_manager.load("detail_system")
    user = prompt_manager.load("feet_analysis_user")
    result = await provider.analyze_image(
        crop.image,
        user.content,
        system_prompt=system.content,
    )
    try:
        payload = FeetDetailPayload.model_validate(result.parsed)
    except ValidationError as exc:
        raise AIInvalidResponseError(
            detail="脚部细化结果字段类型不正确",
        ) from exc

    return FeetObservation(
        payload=payload,
        result=result,
        prompt_versions={
            "detail_system": system.version,
            "feet_analysis_user": user.version,
        },
        sock_geometries={
            Side.LEFT: _geometry(
                payload.left_foot.socks,
                crop,
                source_width,
                source_height,
            ),
            Side.RIGHT: _geometry(
                payload.right_foot.socks,
                crop,
                source_width,
                source_height,
            ),
        },
        shoe_geometries={
            Side.LEFT: _geometry(
                payload.left_foot.shoes,
                crop,
                source_width,
                source_height,
            ),
            Side.RIGHT: _geometry(
                payload.right_foot.shoes,
                crop,
                source_width,
                source_height,
            ),
        },
        warnings=warnings(payload.warnings),
    )
