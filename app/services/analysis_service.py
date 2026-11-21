"""人物属性分析服务。

该模块负责：
- 组装 Prompt 并调用视觉模型 Provider。
- 严格校验和归一化模型 JSON，所有缺失或无法解析的属性回退为 ``unknown``。
- 统一将模型异常转换为业务异常。
- 预留二期“人物局部裁剪二次分析”的扩展点。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from app.ai.base import BaseVisionProvider, ImageInput, ModelResult
from app.core.config import Settings
from app.core.exceptions import (
    AIInvalidResponseError,
    AppError,
    NoPersonDetectedError,
)
from app.prompts.person_analysis import (
    PERSON_ANALYSIS_SYSTEM_PROMPT,
    build_person_analysis_prompt,
)
from app.schemas.analysis import (
    AnalysisData,
    BoundingBox,
    FeetAttributes,
    HeadAttributes,
    ImageInfo,
    LegAttributes,
    LowerBodyAttributes,
    PersonAnalysis,
    PersonConfidence,
    Position,
    TriState,
    UpperBodyAttributes,
    Visibility,
)

logger = logging.getLogger(__name__)

_MISSING = object()
_UNKNOWN_TEXTS = {
    "",
    "unknown",
    "none",
    "null",
    "n/a",
    "na",
    "undefined",
    "未知",
    "无法判断",
    "无法识别",
    "不清楚",
    "不确定",
}


def _as_tristate(value: Any) -> TriState:
    """把模型返回的布尔/字符串统一转换为三态字符串。"""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and value in (0, 1):
        return "true" if value == 1 else "false"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "是"}:
            return "true"
        if normalized in {"false", "no", "n", "0", "否"}:
            return "false"
    return "unknown"


def _as_text(value: Any, *, max_length: int = 120) -> str:
    """归一化文本属性，无法识别或空值返回 unknown。"""

    if not isinstance(value, str):
        return "unknown"
    text = value.strip().strip('"').strip("'").strip()
    if text.lower() in _UNKNOWN_TEXTS or text in _UNKNOWN_TEXTS:
        return "unknown"
    text = " ".join(text.split())
    return text[:max_length] if text else "unknown"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(number):
        return None
    return number


def _as_confidence(value: Any) -> float | None:
    number = _as_number(value)
    if number is None:
        return None
    return max(0.0, min(1.0, round(number, 4)))


def _as_bbox(value: Any, image_width: int, image_height: int) -> BoundingBox | None:
    """解析 bbox，支持归一化坐标或像素坐标。"""

    if isinstance(value, Mapping):
        raw = [
            value.get("x"),
            value.get("y"),
            value.get("width", value.get("w")),
            value.get("height", value.get("h")),
        ]
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        raw = list(value)
    else:
        return None

    numbers = [_as_number(item) for item in raw]
    if any(item is None for item in numbers):
        return None
    x, y, width, height = (float(item) for item in numbers if item is not None)
    if width <= 0 or height <= 0 or x < 0 or y < 0:
        return None

    if all(0 <= item <= 1.5 for item in (x, y, width, height)):
        if x + width > 1.05 or y + height > 1.05:
            return None
    else:
        if image_width <= 0 or image_height <= 0:
            return None
        if x + width > image_width * 1.01 or y + height > image_height * 1.01:
            return None
        x, y = x / image_width, y / image_height
        width, height = width / image_width, height / image_height

    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    width = min(width, 1.0 - x)
    height = min(height, 1.0 - y)
    if width <= 0 or height <= 0:
        return None

    try:
        return BoundingBox(
            x=round(x, 4),
            y=round(y, 4),
            width=round(width, 4),
            height=round(height, 4),
        )
    except ValidationError:
        return None


def _parse_position(
    value: Any,
    image_width: int,
    image_height: int,
) -> Position:
    if isinstance(value, str):
        return Position(description=_as_text(value), bbox=None)
    mapping = _as_mapping(value)
    return Position(
        description=_as_text(mapping.get("description")),
        bbox=_as_bbox(mapping.get("bbox"), image_width, image_height),
    )


def _parse_visibility(value: Any) -> Visibility:
    mapping = _as_mapping(value)
    return Visibility(
        full_body_visible=_as_tristate(mapping.get("full_body_visible")),
        head_visible=_as_tristate(mapping.get("head_visible")),
        upper_body_visible=_as_tristate(mapping.get("upper_body_visible")),
        legs_visible=_as_tristate(mapping.get("legs_visible")),
        feet_visible=_as_tristate(mapping.get("feet_visible")),
    )


def _parse_head(value: Any) -> HeadAttributes:
    mapping = _as_mapping(value)
    return HeadAttributes(
        wearing_hat=_as_tristate(mapping.get("wearing_hat")),
        hat_type=_as_text(mapping.get("hat_type")),
        hat_color=_as_text(mapping.get("hat_color"), max_length=60),
        wearing_glasses=_as_tristate(mapping.get("wearing_glasses")),
    )


def _parse_upper_body(value: Any) -> UpperBodyAttributes:
    mapping = _as_mapping(value)
    return UpperBodyAttributes(
        wearing_top=_as_tristate(mapping.get("wearing_top")),
        top_type=_as_text(mapping.get("top_type")),
        top_color=_as_text(mapping.get("top_color"), max_length=60),
        wearing_jacket=_as_tristate(mapping.get("wearing_jacket")),
        jacket_color=_as_text(mapping.get("jacket_color"), max_length=60),
    )


def _parse_lower_body(value: Any) -> LowerBodyAttributes:
    mapping = _as_mapping(value)
    return LowerBodyAttributes(
        wearing_pants=_as_tristate(mapping.get("wearing_pants")),
        pants_type=_as_text(mapping.get("pants_type")),
        pants_color=_as_text(mapping.get("pants_color"), max_length=60),
        wearing_skirt=_as_tristate(mapping.get("wearing_skirt")),
        skirt_color=_as_text(mapping.get("skirt_color"), max_length=60),
    )


def _parse_legs(value: Any) -> LegAttributes:
    mapping = _as_mapping(value)
    return LegAttributes(
        skin_visible=_as_tristate(mapping.get("skin_visible")),
        wearing_tights=_as_tristate(mapping.get("wearing_tights")),
        wearing_stockings=_as_tristate(mapping.get("wearing_stockings")),
        wearing_long_pants=_as_tristate(mapping.get("wearing_long_pants")),
        wearing_shorts=_as_tristate(mapping.get("wearing_shorts")),
        leg_covering=_as_text(mapping.get("leg_covering")),
    )


def _parse_feet(value: Any) -> FeetAttributes:
    mapping = _as_mapping(value)
    return FeetAttributes(
        wearing_socks=_as_tristate(mapping.get("wearing_socks")),
        socks_type=_as_text(mapping.get("socks_type")),
        socks_color=_as_text(mapping.get("socks_color"), max_length=60),
        wearing_shoes=_as_tristate(mapping.get("wearing_shoes")),
        shoes_type=_as_text(mapping.get("shoes_type")),
        shoes_color=_as_text(mapping.get("shoes_color"), max_length=60),
        barefoot=_as_tristate(mapping.get("barefoot")),
    )


def _parse_confidence(value: Any) -> PersonConfidence:
    mapping = _as_mapping(value)
    return PersonConfidence(
        head=_as_confidence(mapping.get("head")),
        upper_body=_as_confidence(mapping.get("upper_body")),
        lower_body=_as_confidence(mapping.get("lower_body")),
        legs=_as_confidence(mapping.get("legs")),
        feet=_as_confidence(mapping.get("feet")),
    )


def _parse_evidence(value: Any) -> dict[str, str]:
    evidence: dict[str, str] = {}
    if not isinstance(value, Mapping):
        return evidence
    for raw_key, raw_value in list(value.items())[:20]:
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            continue
        key = raw_key.strip()[:50]
        text = " ".join(raw_value.split())[:200]
        if key and text:
            evidence[key] = text
    return evidence


def _parse_person(
    value: Mapping[str, Any],
    person_id: int,
    image_width: int,
    image_height: int,
) -> PersonAnalysis:
    try:
        return PersonAnalysis(
            person_id=person_id,
            position=_parse_position(
                value.get("position"), image_width, image_height
            ),
            visibility=_parse_visibility(value.get("visibility")),
            head=_parse_head(value.get("head")),
            upper_body=_parse_upper_body(value.get("upper_body")),
            lower_body=_parse_lower_body(value.get("lower_body")),
            legs=_parse_legs(value.get("legs")),
            feet=_parse_feet(value.get("feet")),
            confidence=_parse_confidence(value.get("confidence")),
            evidence=_parse_evidence(value.get("evidence")),
        )
    except ValidationError as exc:
        raise AIInvalidResponseError(
            detail="模型返回的人物字段格式不正确",
        ) from exc


def parse_analysis_payload(
    payload: Mapping[str, Any],
    image_width: int,
    image_height: int,
) -> AnalysisData:
    """把模型 JSON 转换为严格的后端响应结构。

    缺少单个属性时回退为 unknown；缺少 persons 列表或列表结构非法时视为
    模型返回无效，抛出 ``AIInvalidResponseError``。
    """

    persons_raw = payload.get("persons", _MISSING)
    if persons_raw is _MISSING:
        raise AIInvalidResponseError(detail="模型返回结果缺少 persons 字段")
    if not isinstance(persons_raw, list):
        raise AIInvalidResponseError(detail="模型返回的 persons 字段不是数组")

    persons: list[PersonAnalysis] = []
    for index, item in enumerate(persons_raw):
        if not isinstance(item, Mapping):
            raise AIInvalidResponseError(
                detail="模型返回的 persons 数组包含非法元素",
            )
        persons.append(_parse_person(item, index + 1, image_width, image_height))

    if not persons:
        raise NoPersonDetectedError()

    model_count = payload.get("person_count")
    if isinstance(model_count, int) and model_count != len(persons):
        logger.warning(
            "模型返回人物数量与数组长度不一致: person_count=%s actual=%s",
            model_count,
            len(persons),
        )

    return AnalysisData(
        image_info=ImageInfo(
            width=image_width,
            height=image_height,
            person_count=len(persons),
        ),
        persons=persons,
    )


class AnalysisService:
    """图片人物属性分析编排服务。"""

    def __init__(
        self,
        provider: BaseVisionProvider,
        settings: Settings,
    ) -> None:
        self._provider = provider
        self._settings = settings

    async def analyze(
        self,
        image: ImageInput,
        *,
        request_id: str | None = None,
    ) -> AnalysisData:
        """调用视觉模型并返回校验后的人物属性结果。"""

        prompt = build_person_analysis_prompt(image.width, image.height)
        result = await self._provider.analyze_image(
            image,
            prompt,
            system_prompt=PERSON_ANALYSIS_SYSTEM_PROMPT,
        )

        try:
            data = parse_analysis_payload(
                result.parsed,
                image_width=image.width,
                image_height=image.height,
            )
        except AppError:
            raise
        except Exception as exc:
            logger.exception("解析模型分析结果时发生未预期异常")
            raise AIInvalidResponseError() from exc

        self._log_usage(result, len(data.persons), request_id=request_id)

        if self._settings.enable_crop_refinement:
            data = await self.refine_with_crops(image, data)
        return data

    async def refine_with_crops(
        self,
        image: ImageInput,
        data: AnalysisData,
    ) -> AnalysisData:
        """二期局部裁剪分析扩展点。

        计划实现：
        1. 使用首轮 bbox 或检测模型裁出每个人物；
        2. 对头部、上身、腿部、脚部区域分别调用视觉模型；
        3. 按人物编号合并结果，并保留首轮中可信度更高的属性。

        当前版本仅提供接口占位，默认配置 ``ENABLE_CROP_REFINEMENT=false``。
        启用后仍返回首轮结果，不会调用额外模型，避免产生不可控费用。
        """

        logger.warning(
            "ENABLE_CROP_REFINEMENT 已开启，但局部分析尚未实现，返回首轮结果"
        )
        return data

    @staticmethod
    def _log_usage(
        result: ModelResult,
        person_count: int,
        *,
        request_id: str | None = None,
    ) -> None:
        logger.info(
            "分析完成 request_id=%s provider=%s model=%s persons=%s usage=%s",
            request_id,
            result.provider,
            result.model,
            person_count,
            result.usage,
        )
