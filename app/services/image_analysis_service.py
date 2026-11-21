"""图片多阶段分析服务。

流程：
1. 全图概览：人物、朝向、下装、身体区域可观察性。
2. 下装裁剪细化。
3. 腿部裁剪分析丝袜。
4. 脚部裁剪分析普通袜子、鞋子和赤脚。
5. Python 规则引擎完成可观察性、穿戴关系和旧响应兼容。

旧 Provider 如果返回原有 ``persons`` 结构，自动走兼容回退路径，保持旧测试
和旧客户端可用。
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.ai.base import BaseVisionProvider, ImageInput, ModelResult
from app.ai.prompt_manager import PromptManager
from app.analyzers.barefoot_analyzer import resolve_foot_state
from app.analyzers.clothing_analyzer import analyze_clothing
from app.analyzers.feet_analyzer import FeetObservation, analyze_feet
from app.analyzers.hosiery_analyzer import analyze_hosiery
from app.analyzers.items import (
    build_item,
    build_lower_body_item,
    category_from_value,
    not_visible_item,
)
from app.analyzers.parsing import (
    confidence_or,
    enum_or,
    geometry_from_box,
    resolve_person_side,
    side_basis_or,
    visibility_or,
)
from app.analyzers.person_analyzer import (
    LegacyPayloadDetected,
    OverviewObservation,
    analyze_overview,
)
from app.analyzers.shoes_analyzer import build_shoes_item
from app.analyzers.socks_analyzer import build_socks_item
from app.core.config import Settings
from app.core.exceptions import AppError
from app.observability.observability_engine import build_from_payload
from app.processing.image_processor import make_crop
from app.relations.wearing_relation_engine import build_wearing_relations
from app.schemas.analysis import AnalysisData
from app.schemas.analysis_v2 import (
    AnalysisResultV2,
    AnalysisStatus,
    AnalysisWarning,
    BarefootStatus,
    ClothingItem,
    ConfidenceLevel,
    EvidenceSource,
    FootState,
    ItemCategory,
    PersonOrientation,
    Presence,
    Side,
    SideBasis,
    SourceType,
    Visibility,
)
from app.services.analysis_service import AnalysisService, parse_analysis_payload
from app.services.image_service import ProcessedImage
from app.services.legacy_adapter import to_legacy_analysis_data

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ImageAnalysisOutcome:
    data: AnalysisData
    analysis: AnalysisResultV2
    calls: int = 0
    usage: dict[str, int] = field(default_factory=dict)


def _merge_usage(total: dict[str, int], result: ModelResult) -> None:
    for key, value in result.usage.items():
        if isinstance(value, int) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value


def _union_boxes(boxes: list[list[int] | None]) -> list[int] | None:
    valid = [box for box in boxes if box is not None]
    if not valid:
        return None
    return [
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    ]


def _lower_box(
    overview: OverviewObservation,
    source_width: int,
    source_height: int,
) -> list[int] | None:
    box = overview.region_boxes.get("lower_body")
    if box is not None:
        return box
    person = overview.person_bbox
    if person is not None:
        height = person[3] - person[1]
        return [
            person[0],
            person[1] + int(height * 0.45),
            person[2],
            person[3],
        ]
    return [0, int(source_height * 0.4), source_width, source_height]


def _legs_box(
    overview: OverviewObservation,
    source_width: int,
    source_height: int,
) -> list[int] | None:
    box = _union_boxes(
        [
            overview.region_boxes.get("left_leg"),
            overview.region_boxes.get("right_leg"),
        ]
    )
    if box is not None:
        return box
    return _lower_box(overview, source_width, source_height)


def _feet_box(
    overview: OverviewObservation,
    source_width: int,
    source_height: int,
) -> list[int] | None:
    box = _union_boxes(
        [
            overview.region_boxes.get("left_foot"),
            overview.region_boxes.get("right_foot"),
        ]
    )
    if box is not None:
        return box
    person = overview.person_bbox
    if person is not None:
        height = person[3] - person[1]
        return [
            person[0],
            person[1] + int(height * 0.72),
            person[2],
            person[3],
        ]
    return [0, int(source_height * 0.65), source_width, source_height]


def _region_reason(region: Any) -> str:
    if region.occlusion_reason:
        return region.occlusion_reason[0].value
    return "unknown"


def _confidence_rank(confidence: ConfidenceLevel) -> int:
    return {
        ConfidenceLevel.LOW: 1,
        ConfidenceLevel.MEDIUM: 2,
        ConfidenceLevel.HIGH: 3,
    }.get(confidence, 0)


def _uncertain_item(
    *,
    category: ItemCategory,
    side: Side,
    analysis_id: str | None,
    source_id: str | None,
    visibility: Visibility,
) -> ClothingItem:
    return ClothingItem(
        analysis_id=analysis_id,
        source_id=source_id,
        category=category,
        type="unknown",
        side=side,
        image_side=side,
        presence=Presence.UNCERTAIN,
        visibility=visibility,
        confidence=ConfidenceLevel.LOW,
    )


class ImageAnalysisService:
    """图片 V2 多阶段分析编排。"""

    def __init__(
        self,
        provider: BaseVisionProvider,
        legacy_service: AnalysisService,
        settings: Settings,
        *,
        prompt_manager: PromptManager | None = None,
    ) -> None:
        self._provider = provider
        self._legacy = legacy_service
        self._settings = settings
        self._prompts = prompt_manager or PromptManager(settings.prompts_path)

    async def analyze(
        self,
        processed: ProcessedImage,
        *,
        filename: str | None = None,
        request_id: str | None = None,
        max_calls: int | None = None,
        include_clothing_detail: bool = True,
    ) -> ImageAnalysisOutcome:
        analysis_id = uuid.uuid4().hex
        started = time.perf_counter()
        source_id = processed.sha256
        source_width = processed.source_width
        source_height = processed.source_height
        image = ImageInput(
            data=processed.data,
            mime_type=processed.mime_type,
            width=processed.processed_width,
            height=processed.processed_height,
            filename=filename,
        )

        usage: dict[str, int] = {}
        try:
            overview = await analyze_overview(
                self._provider,
                image,
                self._prompts,
                source_width=source_width,
                source_height=source_height,
            )
        except LegacyPayloadDetected as legacy:
            _merge_usage(usage, legacy.result)
            return self._legacy_fallback(
                legacy.result,
                processed,
                analysis_id=analysis_id,
                source_id=source_id,
                prompt_versions=legacy.prompt_versions,
                usage=usage,
                request_id=request_id,
            )

        _merge_usage(usage, overview.result)
        calls = 1
        payload = overview.payload
        if payload.person_count > 1:
            return await self._multi_person_fallback(
                image,
                overview,
                processed,
                analysis_id=analysis_id,
                source_id=source_id,
                usage=usage,
                calls=calls,
                request_id=request_id,
            )

        items: list[ClothingItem] = []
        warnings: list[AnalysisWarning] = [
            AnalysisWarning(code="OVERVIEW_WARNING", message=text)
            for text in overview.warnings
        ]
        prompt_versions = dict(overview.prompt_versions)
        reasoning_warnings: list[str] = []
        budget = max(
            1,
            min(
                max_calls or self._settings.ai_max_calls_per_analysis,
                self._settings.ai_max_calls_per_analysis,
            ),
        )
        side_basis = side_basis_or(payload.side_basis, SideBasis.UNKNOWN)
        side_confidence = confidence_or(
            payload.side_assignment_confidence,
            ConfidenceLevel.LOW,
        )
        side_unknown_from_detail = False

        def observe_side(payload_for_side: Any, fallback: Side) -> None:
            nonlocal side_basis, side_confidence, side_unknown_from_detail
            if not any(
                getattr(payload_for_side, field, None) is not None
                for field in ("person_side", "side_basis", "side_confidence")
            ):
                return
            resolved, confidence, basis = resolve_person_side(
                payload_for_side,
                fallback,
            )
            if resolved is Side.UNKNOWN:
                side_unknown_from_detail = True
                return
            if _confidence_rank(confidence) > _confidence_rank(side_confidence):
                side_confidence = confidence
            if basis is not SideBasis.UNKNOWN:
                side_basis = basis

        lower_payload = payload.lower_body
        lower_geometry = geometry_from_box(
            lower_payload.bbox,
            source_width=source_width,
            source_height=source_height,
            method="overview_estimate",
        )
        lower_item = build_lower_body_item(
            lower_payload,
            analysis_id=analysis_id,
            source_id=source_id,
            geometry=lower_geometry,
            source=EvidenceSource.OVERVIEW,
            style_tags=lower_payload.style_tags,
            design_features=lower_payload.design_features,
            fallback_observation="全图概览中的下装观察",
        )
        items.append(lower_item)

        lower_visibility = visibility_or(
            payload.body_observability.lower_body.visibility,
            Visibility.NOT_VISIBLE,
        )
        detail_attempts = 0
        detail_failed = False
        if (
            include_clothing_detail
            and calls < budget
            and lower_visibility is not Visibility.NOT_VISIBLE
        ):
            crop = make_crop(
                processed,
                "lower_body",
                source_bbox=_lower_box(overview, source_width, source_height),
                padding_ratio=self._settings.analysis_crop_padding_ratio,
                min_side=self._settings.analysis_crop_min_side,
                max_side=self._settings.analysis_crop_max_side,
            )
            if crop is not None:
                detail_attempts += 1
                try:
                    clothing = await analyze_clothing(
                        self._provider,
                        crop,
                        self._prompts,
                        source_width=source_width,
                        source_height=source_height,
                    )
                except AppError as exc:
                    detail_failed = True
                    warnings.append(
                        AnalysisWarning(
                            code=exc.code,
                            message="下装局部细化失败，保留全图结果",
                            detail=exc.detail,
                        )
                    )
                else:
                    calls += 1
                    _merge_usage(usage, clothing.result)
                    prompt_versions.update(clothing.prompt_versions)
                    warnings.extend(
                        AnalysisWarning(code="CLOTHING_WARNING", message=text)
                        for text in clothing.warnings
                    )
                    if (
                        category_from_value(clothing.payload.category)
                        is not ItemCategory.UNKNOWN
                    ):
                        lower_item = build_lower_body_item(
                            clothing.payload,
                            analysis_id=analysis_id,
                            source_id=source_id,
                            geometry=clothing.geometry or lower_geometry,
                            source=EvidenceSource.CROP,
                            style_tags=clothing.payload.style_tags,
                            design_features=clothing.payload.design_features,
                            fallback_observation="下装局部裁剪观察",
                        )
                        items[0] = lower_item

        observability = build_from_payload(payload, items)
        hosiery_observation = None
        feet_observation: FeetObservation | None = None

        legs_visible = any(
            visibility in (Visibility.CLEAR, Visibility.PARTIAL)
            for visibility in (
                observability.left_leg.visibility,
                observability.right_leg.visibility,
            )
        )
        if calls < budget and legs_visible:
            crop = make_crop(
                processed,
                "legs",
                source_bbox=_legs_box(overview, source_width, source_height),
                padding_ratio=self._settings.analysis_crop_padding_ratio,
                min_side=self._settings.analysis_crop_min_side,
                max_side=self._settings.analysis_crop_max_side,
            )
            if crop is not None:
                detail_attempts += 1
                try:
                    hosiery_observation = await analyze_hosiery(
                        self._provider,
                        crop,
                        self._prompts,
                        source_width=source_width,
                        source_height=source_height,
                    )
                except AppError as exc:
                    detail_failed = True
                    warnings.append(
                        AnalysisWarning(
                            code=exc.code,
                            message="丝袜局部分析失败",
                            detail=exc.detail,
                        )
                    )
                else:
                    calls += 1
                    _merge_usage(usage, hosiery_observation.result)
                    prompt_versions.update(hosiery_observation.prompt_versions)
                    warnings.extend(
                        AnalysisWarning(code="HOSIERY_WARNING", message=text)
                        for text in hosiery_observation.warnings
                    )
                    for side in (Side.LEFT, Side.RIGHT):
                        side_payload = hosiery_observation.side_payload(side)
                        if side_payload is not None:
                            observe_side(side_payload, side)
                        region = (
                            observability.left_leg
                            if side is Side.LEFT
                            else observability.right_leg
                        )
                        if side_payload is None:
                            if region.visibility in (
                                Visibility.CLEAR,
                                Visibility.PARTIAL,
                            ):
                                item = _uncertain_item(
                                    category=ItemCategory.HOSIERY,
                                    side=side,
                                    analysis_id=analysis_id,
                                    source_id=source_id,
                                    visibility=region.visibility,
                                )
                            else:
                                item = not_visible_item(
                                    category=ItemCategory.HOSIERY,
                                    item_type="unknown",
                                    side=side,
                                    analysis_id=analysis_id,
                                    source_id=source_id,
                                    reason=_region_reason(region),
                                )
                        else:
                            item = build_item(
                                side_payload,
                                category=ItemCategory.HOSIERY,
                                side=side,
                                analysis_id=analysis_id,
                                source_id=source_id,
                                geometry=hosiery_observation.geometries.get(side),
                                fallback_observation="腿部裁剪图中的丝袜观察",
                            )
                        items.append(item)

        if hosiery_observation is None:
            for side in (Side.LEFT, Side.RIGHT):
                region = (
                    observability.left_leg
                    if side is Side.LEFT
                    else observability.right_leg
                )
                if region.visibility in (Visibility.CLEAR, Visibility.PARTIAL):
                    items.append(
                        _uncertain_item(
                            category=ItemCategory.HOSIERY,
                            side=side,
                            analysis_id=analysis_id,
                            source_id=source_id,
                            visibility=region.visibility,
                        )
                    )
                    reasoning_warnings.append(
                        f"{side.value} 腿可观察，但丝袜细化未完成"
                    )
                else:
                    items.append(
                        not_visible_item(
                            category=ItemCategory.HOSIERY,
                            item_type="unknown",
                            side=side,
                            analysis_id=analysis_id,
                            source_id=source_id,
                            reason=_region_reason(region),
                        )
                    )

        feet_visible = any(
            visibility in (Visibility.CLEAR, Visibility.PARTIAL)
            for visibility in (
                observability.left_foot.visibility,
                observability.right_foot.visibility,
            )
        )
        if calls < budget and feet_visible:
            crop = make_crop(
                processed,
                "feet",
                source_bbox=_feet_box(overview, source_width, source_height),
                padding_ratio=self._settings.analysis_crop_padding_ratio,
                min_side=self._settings.analysis_crop_min_side,
                max_side=self._settings.analysis_crop_max_side,
            )
            if crop is not None:
                detail_attempts += 1
                try:
                    feet_observation = await analyze_feet(
                        self._provider,
                        crop,
                        self._prompts,
                        source_width=source_width,
                        source_height=source_height,
                    )
                except AppError as exc:
                    detail_failed = True
                    warnings.append(
                        AnalysisWarning(
                            code=exc.code,
                            message="脚部局部分析失败",
                            detail=exc.detail,
                        )
                    )
                else:
                    calls += 1
                    _merge_usage(usage, feet_observation.result)
                    prompt_versions.update(feet_observation.prompt_versions)
                    warnings.extend(
                        AnalysisWarning(code="FEET_WARNING", message=text)
                        for text in feet_observation.warnings
                    )
                    for side in (Side.LEFT, Side.RIGHT):
                        side_payload = feet_observation.side_payload(side)
                        observe_side(side_payload, side)
                        items.append(
                            build_socks_item(
                                side_payload,
                                side=side,
                                analysis_id=analysis_id,
                                source_id=source_id,
                                geometry=feet_observation.sock_geometries.get(side),
                                occlusion_reason=_region_reason(
                                    observability.left_foot
                                    if side is Side.LEFT
                                    else observability.right_foot
                                ),
                            )
                        )
                        items.append(
                            build_shoes_item(
                                side_payload,
                                side=side,
                                analysis_id=analysis_id,
                                source_id=source_id,
                                geometry=feet_observation.shoe_geometries.get(side),
                                occlusion_reason=_region_reason(
                                    observability.left_foot
                                    if side is Side.LEFT
                                    else observability.right_foot
                                ),
                            )
                        )

        if feet_observation is None:
            for side in (Side.LEFT, Side.RIGHT):
                region = (
                    observability.left_foot
                    if side is Side.LEFT
                    else observability.right_foot
                )
                for category in (ItemCategory.SOCKS, ItemCategory.SHOES):
                    if region.visibility in (Visibility.CLEAR, Visibility.PARTIAL):
                        items.append(
                            _uncertain_item(
                                category=category,
                                side=side,
                                analysis_id=analysis_id,
                                source_id=source_id,
                                visibility=region.visibility,
                            )
                        )
                    else:
                        items.append(
                            not_visible_item(
                                category=category,
                                item_type="unknown",
                                side=side,
                                analysis_id=analysis_id,
                                source_id=source_id,
                                reason=_region_reason(region),
                            )
                        )
                if region.visibility in (Visibility.CLEAR, Visibility.PARTIAL):
                    reasoning_warnings.append(
                        f"{side.value} 脚可观察，但脚部细化未完成"
                    )

        feet: list[FootState] = []
        if feet_observation is not None:
            for side in (Side.LEFT, Side.RIGHT):
                body_region = (
                    observability.left_foot
                    if side is Side.LEFT
                    else observability.right_foot
                )
                state, foot_warnings = resolve_foot_state(
                    side=side,
                    side_payload=feet_observation.side_payload(side),
                    body_visibility=body_region.visibility,
                    items=items,
                )
                feet.append(state)
                warnings.extend(
                    AnalysisWarning(code="BAREFOOT_WARNING", message=text)
                    for text in foot_warnings
                )
        else:
            for side in (Side.LEFT, Side.RIGHT):
                body_region = (
                    observability.left_foot
                    if side is Side.LEFT
                    else observability.right_foot
                )
                if body_region.visibility in (
                    Visibility.CLEAR,
                    Visibility.PARTIAL,
                ):
                    feet.append(
                        FootState(
                            side=side,
                            image_side=side,
                            barefoot=BarefootStatus.UNKNOWN,
                            visibility=body_region.visibility,
                            uncertainty_reason="脚部可观察，但脚部细化未完成",
                        )
                    )
                else:
                    feet.append(
                        FootState(
                            side=side,
                            image_side=side,
                            barefoot=BarefootStatus.NOT_VISIBLE,
                            visibility=body_region.visibility,
                            uncertainty_reason="脚部不可见",
                        )
                    )

        if detail_attempts == 0:
            reasoning_warnings.append("没有可用的局部裁剪分析")

        warnings.extend(
            AnalysisWarning(code="ANALYSIS_LIMITATION", message=text)
            for text in reasoning_warnings
        )
        if side_unknown_from_detail:
            side_basis = SideBasis.UNKNOWN
            side_confidence = ConfidenceLevel.LOW
            warnings.append(
                AnalysisWarning(
                    code="SIDE_ASSIGNMENT_UNKNOWN",
                    message="局部裁剪无法确定人物自身左右，相关物品 side 使用 unknown",
                )
            )
        relations = build_wearing_relations(items)
        analysis = AnalysisResultV2(
            analysis_id=analysis_id,
            source_id=source_id,
            source_type=SourceType.IMAGE,
            status=AnalysisStatus.PARTIAL if detail_failed else AnalysisStatus.COMPLETED,
            person_count=1,
            person_orientation=enum_or(
                payload.person_orientation,
                PersonOrientation,
                PersonOrientation.UNKNOWN,
            ),
            side_basis=side_basis,
            side_assignment_confidence=side_confidence,
            items=items,
            body_observability=observability,
            feet=feet,
            wearing_relations=relations,
            warnings=warnings,
            prompt_versions=prompt_versions,
            model=overview.result.model,
            usage=usage,
        )
        legacy_data = to_legacy_analysis_data(
            analysis,
            width=source_width,
            height=source_height,
        )
        logger.info(
            "图片V2分析完成 request_id=%s analysis_id=%s calls=%s items=%s "
            "duration_ms=%.1f usage=%s",
            request_id,
            analysis_id,
            calls,
            len(items),
            (time.perf_counter() - started) * 1000,
            usage,
        )
        return ImageAnalysisOutcome(
            data=legacy_data,
            analysis=analysis,
            calls=calls,
            usage=usage,
        )

    def _legacy_fallback(
        self,
        result: ModelResult,
        processed: ProcessedImage,
        *,
        analysis_id: str,
        source_id: str,
        prompt_versions: dict[str, str],
        usage: dict[str, int],
        request_id: str | None,
    ) -> ImageAnalysisOutcome:
        started = time.perf_counter()
        data = parse_analysis_payload(
            result.parsed,
            image_width=processed.source_width,
            image_height=processed.source_height,
        )
        analysis = AnalysisResultV2(
            analysis_id=analysis_id,
            source_id=source_id,
            source_type=SourceType.IMAGE,
            status=AnalysisStatus.UNSUPPORTED_MULTI_PERSON
            if len(data.persons) > 1
            else AnalysisStatus.COMPLETED,
            person_count=len(data.persons),
            person_orientation=PersonOrientation.UNKNOWN,
            warnings=[
                AnalysisWarning(
                    code="LEGACY_RESPONSE_FALLBACK",
                    message="模型返回旧版人物结构，已使用兼容路径",
                )
            ],
            prompt_versions=prompt_versions,
            model=result.model,
            usage=usage,
        )
        data = data.model_copy(
            update={
                "schema_version": 2,
                "analysis_id": analysis.analysis_id,
                "person_count": analysis.person_count,
                "source_type": analysis.source_type,
                "status": analysis.status,
                "person_orientation": analysis.person_orientation,
                "side_basis": analysis.side_basis,
                "side_assignment_confidence": analysis.side_assignment_confidence,
                "items": analysis.items,
                "body_observability": analysis.body_observability,
                "feet": analysis.feet,
                "wearing_relations": analysis.wearing_relations,
                "evidence": analysis.evidence,
                "warnings": analysis.warnings,
                "created_at": analysis.created_at,
                "prompt_versions": analysis.prompt_versions,
                "model": analysis.model,
                "usage": analysis.usage,
            }
        )
        logger.info(
            "图片V2分析使用旧结构兼容路径 request_id=%s analysis_id=%s persons=%s "
            "duration_ms=%.1f",
            request_id,
            analysis_id,
            len(data.persons),
            (time.perf_counter() - started) * 1000,
        )
        return ImageAnalysisOutcome(
            data=data,
            analysis=analysis,
            calls=1,
            usage=usage,
        )

    async def _multi_person_fallback(
        self,
        image: ImageInput,
        overview: OverviewObservation,
        processed: ProcessedImage,
        *,
        analysis_id: str,
        source_id: str,
        usage: dict[str, int],
        calls: int,
        request_id: str | None,
    ) -> ImageAnalysisOutcome:
        started = time.perf_counter()
        data = await self._legacy.analyze(image, request_id=request_id)
        analysis = AnalysisResultV2(
            analysis_id=analysis_id,
            source_id=source_id,
            source_type=SourceType.IMAGE,
            status=AnalysisStatus.UNSUPPORTED_MULTI_PERSON,
            person_count=len(data.persons),
            person_orientation=enum_or(
                overview.payload.person_orientation,
                PersonOrientation,
                PersonOrientation.UNKNOWN,
            ),
            warnings=[
                AnalysisWarning(
                    code="MULTI_PERSON_V2_UNSUPPORTED",
                    message="当前 V2 物品分析仅支持单人图片，已保留旧版多人结果",
                )
            ],
            prompt_versions=dict(overview.prompt_versions),
            model=overview.result.model,
            usage=usage,
        )
        data = data.model_copy(
            update={
                "schema_version": 2,
                "analysis_id": analysis.analysis_id,
                "person_count": analysis.person_count,
                "source_type": analysis.source_type,
                "status": analysis.status,
                "person_orientation": analysis.person_orientation,
                "side_basis": analysis.side_basis,
                "side_assignment_confidence": analysis.side_assignment_confidence,
                "items": analysis.items,
                "body_observability": analysis.body_observability,
                "feet": analysis.feet,
                "wearing_relations": analysis.wearing_relations,
                "evidence": analysis.evidence,
                "warnings": analysis.warnings,
                "created_at": analysis.created_at,
                "prompt_versions": analysis.prompt_versions,
                "model": analysis.model,
                "usage": analysis.usage,
            }
        )
        logger.info(
            "图片V2多人回退 request_id=%s analysis_id=%s persons=%s duration_ms=%.1f",
            request_id,
            analysis_id,
            len(data.persons),
            (time.perf_counter() - started) * 1000,
        )
        return ImageAnalysisOutcome(
            data=data,
            analysis=analysis,
            calls=calls + 1,
            usage=usage,
        )
