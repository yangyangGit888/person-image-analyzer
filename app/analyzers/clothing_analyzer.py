"""下装局部细化分析。"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from app.ai.base import BaseVisionProvider, ModelResult
from app.ai.prompt_manager import PromptManager
from app.analyzers.models import ClothingDetailPayload
from app.analyzers.parsing import geometry_from_box, warnings
from app.core.exceptions import AIInvalidResponseError
from app.processing.image_processor import CropRegion
from app.schemas.analysis_v2 import Geometry


@dataclass(slots=True)
class ClothingObservation:
    payload: ClothingDetailPayload
    result: ModelResult
    prompt_versions: dict[str, str]
    geometry: Geometry | None = None
    warnings: list[str] = field(default_factory=list)


async def analyze_clothing(
    provider: BaseVisionProvider,
    crop: CropRegion,
    prompt_manager: PromptManager,
    *,
    source_width: int,
    source_height: int,
) -> ClothingObservation:
    system = prompt_manager.load("detail_system")
    user = prompt_manager.load("clothing_detail_user")
    result = await provider.analyze_image(
        crop.image,
        user.content,
        system_prompt=system.content,
    )
    try:
        payload = ClothingDetailPayload.model_validate(result.parsed)
    except ValidationError as exc:
        raise AIInvalidResponseError(
            detail="下装细化结果字段类型不正确",
        ) from exc

    geometry = geometry_from_box(
        payload.bbox,
        source_width=source_width,
        source_height=source_height,
        crop=crop,
        method="crop_estimate",
    )
    return ClothingObservation(
        payload=payload,
        result=result,
        prompt_versions={
            "detail_system": system.version,
            "clothing_detail_user": user.version,
        },
        geometry=geometry,
        warnings=warnings(payload.warnings),
    )
