"""左右腿丝袜局部分析。"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from app.ai.base import BaseVisionProvider, ModelResult
from app.ai.prompt_manager import PromptManager
from app.analyzers.models import LegsDetailPayload, SideItemPayload
from app.analyzers.parsing import (
    geometry_from_box,
    normalized_token,
    optional_bool,
    warnings,
)
from app.core.exceptions import AIInvalidResponseError
from app.processing.image_processor import CropRegion
from app.schemas.analysis_v2 import Geometry, Side


@dataclass(slots=True)
class HosieryObservation:
    payload: LegsDetailPayload
    result: ModelResult
    prompt_versions: dict[str, str]
    geometries: dict[Side, Geometry | None] = field(default_factory=dict)
    pair_connection_visible: bool | None = None
    warnings: list[str] = field(default_factory=list)

    def side_payload(self, side: Side) -> SideItemPayload | None:
        return self.payload.left_leg if side is Side.LEFT else self.payload.right_leg


def _geometry(
    payload: SideItemPayload | None,
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


async def analyze_hosiery(
    provider: BaseVisionProvider,
    crop: CropRegion,
    prompt_manager: PromptManager,
    *,
    source_width: int,
    source_height: int,
) -> HosieryObservation:
    system = prompt_manager.load("detail_system")
    user = prompt_manager.load("legs_analysis_user")
    result = await provider.analyze_image(
        crop.image,
        user.content,
        system_prompt=system.content,
    )
    try:
        payload = LegsDetailPayload.model_validate(result.parsed)
    except ValidationError as exc:
        raise AIInvalidResponseError(
            detail="丝袜细化结果字段类型不正确",
        ) from exc

    pair_connection = optional_bool(payload.pair_connection_visible)
    prompt_warnings = warnings(payload.warnings)

    # 仅凭长度不能判断连裤袜；没有腰部/裆部/连接证据时降级为 unknown。
    for side in (Side.LEFT, Side.RIGHT):
        side_payload = payload.left_leg if side is Side.LEFT else payload.right_leg
        if side_payload is None:
            continue
        if normalized_token(side_payload.type) == "pantyhose" and pair_connection is not True:
            side_payload.type = "unknown"
            prompt_warnings.append(
                f"{side.value} 侧丝袜缺少连裤袜连接证据，类型降级为 unknown"
            )

    return HosieryObservation(
        payload=payload,
        result=result,
        prompt_versions={
            "detail_system": system.version,
            "legs_analysis_user": user.version,
        },
        geometries={
            Side.LEFT: _geometry(payload.left_leg, crop, source_width, source_height),
            Side.RIGHT: _geometry(payload.right_leg, crop, source_width, source_height),
        },
        pair_connection_visible=pair_connection,
        warnings=prompt_warnings,
    )
