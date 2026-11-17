"""基于 DeepSeek 视觉结果的可编辑区域定位与存在性判断。"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from app.ai.base import BaseVisionProvider, ImageInput
from app.core.exceptions import (
    AIEmptyResponseError,
    AIInvalidJSONError,
    AIInvalidResponseError,
)
from app.prompts.edit_regions import (
    EDIT_REGION_SYSTEM_PROMPT,
    build_edit_region_prompt,
)
from app.schemas.analysis import PersonAnalysis, TriState
from app.schemas.edit import ExistenceStatus, PixelBox, RegionType

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DetectedRegion:
    region_type: RegionType
    existence_status: ExistenceStatus
    bbox: PixelBox | None
    confidence: float | None
    attributes: dict[str, str] = field(default_factory=dict)
    detection_method: str = "deepseek"


class RegionDetectionService:
    def __init__(self, provider: BaseVisionProvider) -> None:
        self._provider = provider

    async def detect(
        self,
        image: ImageInput,
        persons: list[PersonAnalysis],
        person_boxes: list[PixelBox],
    ) -> list[list[DetectedRegion]]:
        """返回与 persons 顺序一致的、每人 7 个可编辑区域。"""

        prompt_persons = []
        for index, box in enumerate(person_boxes, start=1):
            prompt_persons.append(
                {
                    "person_index": index,
                    "bbox": {
                        "x": round(box.x / image.width, 4),
                        "y": round(box.y / image.height, 4),
                        "width": round(box.width / image.width, 4),
                        "height": round(box.height / image.height, 4),
                    },
                }
            )
        prompt = build_edit_region_prompt(
            image.width,
            image.height,
            prompt_persons,
        )
        raw_regions: list[dict[str, Any]] = []
        try:
            result = await self._provider.analyze_image(
                image,
                prompt,
                system_prompt=EDIT_REGION_SYSTEM_PROMPT,
            )
            raw_persons = result.parsed.get("persons")
            if isinstance(raw_persons, list):
                raw_regions = [
                    item if isinstance(item, dict) else {}
                    for item in raw_persons
                ]
        except (AIInvalidJSONError, AIInvalidResponseError, AIEmptyResponseError) as exc:
            logger.warning("区域定位结果无效，使用人物分析结果降级: %s", type(exc).__name__)
        except Exception:
            # 鉴权、超时等错误交由上层统一处理；区域缺失时可降级。
            logger.warning("区域定位模型调用失败，使用人物分析结果降级")
            raw_regions = []

        by_index: dict[int, dict[str, Any]] = {}
        for fallback_index, item in enumerate(raw_regions, start=1):
            index = item.get("person_index")
            if not isinstance(index, int) or isinstance(index, bool):
                index = fallback_index
            by_index[index] = item

        results: list[list[DetectedRegion]] = []
        for person_index, (person, box) in enumerate(
            zip(persons, person_boxes, strict=True),
            start=1,
        ):
            raw_person = by_index.get(person_index, {})
            raw_region_map = raw_person.get("regions")
            region_map = raw_region_map if isinstance(raw_region_map, dict) else {}
            results.append(
                self._build_person_regions(
                    person,
                    box,
                    image.width,
                    image.height,
                    region_map,
                )
            )
        return results

    def _build_person_regions(
        self,
        person: PersonAnalysis,
        person_box: PixelBox,
        image_width: int,
        image_height: int,
        raw_regions: dict[str, Any],
    ) -> list[DetectedRegion]:
        attribute_defaults: dict[RegionType, tuple[TriState, dict[str, str]]] = {
            RegionType.HAT: (
                person.head.wearing_hat,
                {
                    "type": person.head.hat_type,
                    "color": person.head.hat_color,
                },
            ),
            RegionType.GLASSES: (
                person.head.wearing_glasses,
                {"type": "unknown", "color": "unknown"},
            ),
            RegionType.HAIR: (
                (
                    "true"
                    if person.visibility.head_visible == "true"
                    else "unknown"
                ),
                {"type": "unknown", "color": "unknown"},
            ),
            RegionType.EARRING: ("unknown", {"type": "unknown", "color": "unknown"}),
            RegionType.TOP: (
                person.upper_body.wearing_top,
                {
                    "type": person.upper_body.top_type,
                    "color": person.upper_body.top_color,
                },
            ),
            RegionType.PANTS: (
                person.lower_body.wearing_pants,
                {
                    "type": person.lower_body.pants_type,
                    "color": person.lower_body.pants_color,
                },
            ),
            RegionType.SHOES: (
                person.feet.wearing_shoes,
                {
                    "type": person.feet.shoes_type,
                    "color": person.feet.shoes_color,
                },
            ),
        }

        detected: list[DetectedRegion] = []
        for region_type in (
            RegionType.HAT,
            RegionType.GLASSES,
            RegionType.HAIR,
            RegionType.EARRING,
            RegionType.TOP,
            RegionType.PANTS,
            RegionType.SHOES,
        ):
            raw = raw_regions.get(region_type.value)
            raw = raw if isinstance(raw, dict) else {}
            existence = self._parse_existence(raw.get("existence"))
            confidence = self._parse_confidence(raw.get("confidence"))
            bbox = self._parse_bbox(
                raw.get("bbox"),
                image_width,
                image_height,
            )
            attributes = {
                "type": self._parse_text(raw.get("type")),
                "color": self._parse_text(raw.get("color")),
            }

            default_tri_state, default_attributes = attribute_defaults[region_type]
            if existence is ExistenceStatus.UNKNOWN:
                existence = self._from_tristate(default_tri_state)
            for key, value in default_attributes.items():
                if attributes[key] == "unknown" and value != "unknown":
                    attributes[key] = value

            if existence is ExistenceStatus.MISSING:
                bbox = None
            elif existence is ExistenceStatus.EXISTING and bbox is None:
                bbox = self._heuristic_bbox(region_type, person_box, image_width, image_height)

            detected.append(
                DetectedRegion(
                    region_type=region_type,
                    existence_status=existence,
                    bbox=bbox,
                    confidence=confidence,
                    attributes=attributes,
                    detection_method="deepseek" if raw else "heuristic",
                )
            )
        return detected

    @staticmethod
    def _from_tristate(value: TriState) -> ExistenceStatus:
        if value == "true":
            return ExistenceStatus.EXISTING
        if value == "false":
            return ExistenceStatus.MISSING
        return ExistenceStatus.UNKNOWN

    @staticmethod
    def _parse_existence(value: Any) -> ExistenceStatus:
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"existing", "present", "true", "是", "存在"}:
                return ExistenceStatus.EXISTING
            if text in {"missing", "absent", "false", "否", "不存在", "没有"}:
                return ExistenceStatus.MISSING
        return ExistenceStatus.UNKNOWN

    @staticmethod
    def _parse_confidence(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return max(0.0, min(1.0, float(value)))
        return None

    @staticmethod
    def _parse_text(value: Any) -> str:
        if not isinstance(value, str):
            return "unknown"
        text = " ".join(value.split()).strip()
        if not text or text.lower() in {"none", "null", "unknown", "未知", "无法判断"}:
            return "unknown"
        return text[:80]

    @staticmethod
    def _parse_bbox(
        value: Any,
        image_width: int,
        image_height: int,
    ) -> PixelBox | None:
        if not isinstance(value, dict):
            return None
        try:
            x = float(value["x"])
            y = float(value["y"])
            width = float(value["width"])
            height = float(value["height"])
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(item) for item in (x, y, width, height)):
            return None
        if width <= 0 or height <= 0:
            return None
        if all(0 <= item <= 1.5 for item in (x, y, width, height)):
            x, width = x * image_width, width * image_width
            y, height = y * image_height, height * image_height
        x1 = max(0, round(x))
        y1 = max(0, round(y))
        x2 = min(image_width, round(x + width))
        y2 = min(image_height, round(y + height))
        if x2 <= x1 or y2 <= y1:
            return None
        return PixelBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    @staticmethod
    def _heuristic_bbox(
        region_type: RegionType,
        person: PixelBox,
        image_width: int,
        image_height: int,
    ) -> PixelBox | None:
        ratios = {
            RegionType.HAT: (0.12, 0.0, 0.88, 0.18),
            RegionType.GLASSES: (0.2, 0.1, 0.8, 0.23),
            RegionType.HAIR: (0.12, 0.0, 0.88, 0.25),
            RegionType.EARRING: (0.04, 0.14, 0.96, 0.3),
            RegionType.TOP: (0.12, 0.2, 0.88, 0.58),
            RegionType.PANTS: (0.2, 0.5, 0.8, 0.94),
            RegionType.SHOES: (0.18, 0.9, 0.82, 1.0),
        }
        left, top, right, bottom = ratios[region_type]
        x1 = max(0, person.x + int(person.width * left))
        y1 = max(0, person.y + int(person.height * top))
        x2 = min(image_width, person.x + int(person.width * right))
        y2 = min(image_height, person.y + int(person.height * bottom))
        if x2 <= x1 or y2 <= y1:
            return None
        return PixelBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
