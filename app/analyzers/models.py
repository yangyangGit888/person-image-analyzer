"""模型原始响应的宽松 Pydantic 容器。

模型输出可能存在缺失字段、类型不一致或未知枚举值。这里只约束顶层结构，
具体字段由 ``parsing.py`` 做安全归一化，避免因为一个可选字段导致整次分析失败。
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.schemas.analysis_v2 import SchemaBaseV2


class OverviewRegionPayload(SchemaBaseV2):
    visibility: Any = "not_visible"
    occlusion_reason: list[Any] = Field(default_factory=list)
    bbox: Any = None
    confidence: Any = "low"


class OverviewBodyPayload(SchemaBaseV2):
    lower_body: OverviewRegionPayload = Field(default_factory=OverviewRegionPayload)
    left_leg: OverviewRegionPayload = Field(default_factory=OverviewRegionPayload)
    right_leg: OverviewRegionPayload = Field(default_factory=OverviewRegionPayload)
    left_foot: OverviewRegionPayload = Field(default_factory=OverviewRegionPayload)
    right_foot: OverviewRegionPayload = Field(default_factory=OverviewRegionPayload)


class OverviewLowerBodyPayload(SchemaBaseV2):
    category: Any = "unknown"
    type: Any = "unknown"
    presence: Any = "uncertain"
    visibility: Any = "not_visible"
    style_tags: list[Any] = Field(default_factory=list)
    design_features: list[Any] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    bbox: Any = None
    occlusion_reason: list[Any] = Field(default_factory=list)
    confidence: Any = "low"
    evidence: list[Any] = Field(default_factory=list)


class OverviewPayload(SchemaBaseV2):
    person_count: int = Field(ge=0)
    person_orientation: Any = "unknown"
    person_pose: Any = "unknown"
    person_bbox: Any = None
    side_basis: Any = "unknown"
    side_assignment_confidence: Any = "low"
    lower_body: OverviewLowerBodyPayload = Field(
        default_factory=OverviewLowerBodyPayload
    )
    body_observability: OverviewBodyPayload = Field(
        default_factory=OverviewBodyPayload
    )
    warnings: list[Any] = Field(default_factory=list)


class SideItemPayload(SchemaBaseV2):
    person_side: Any = None
    side_basis: Any = None
    side_confidence: Any = None
    presence: Any = "uncertain"
    visibility: Any = "not_visible"
    type: Any = "unknown"
    attributes: dict[str, Any] = Field(default_factory=dict)
    bbox: Any = None
    occlusion_reason: list[Any] = Field(default_factory=list)
    confidence: Any = "low"
    evidence: list[Any] = Field(default_factory=list)


class ClothingDetailPayload(SideItemPayload):
    category: Any = "unknown"
    style_tags: list[Any] = Field(default_factory=list)
    design_features: list[Any] = Field(default_factory=list)
    warnings: list[Any] = Field(default_factory=list)


class LegsDetailPayload(SchemaBaseV2):
    left_leg: SideItemPayload | None = None
    right_leg: SideItemPayload | None = None
    pair_connection_visible: Any = "unknown"
    warnings: list[Any] = Field(default_factory=list)


class FeetSidePayload(SchemaBaseV2):
    person_side: Any = None
    side_basis: Any = None
    side_confidence: Any = None
    visibility: Any = "not_visible"
    barefoot: Any = "unknown"
    skin_visible: Any = "unknown"
    socks: SideItemPayload | None = None
    shoes: SideItemPayload | None = None
    occlusion_reason: list[Any] = Field(default_factory=list)
    confidence: Any = "low"
    evidence: list[Any] = Field(default_factory=list)


class FeetDetailPayload(SchemaBaseV2):
    left_foot: FeetSidePayload = Field(default_factory=FeetSidePayload)
    right_foot: FeetSidePayload = Field(default_factory=FeetSidePayload)
    warnings: list[Any] = Field(default_factory=list)
