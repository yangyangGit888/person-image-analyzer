"""图片分析相关的 Pydantic 数据模型。

三态属性统一使用字符串 ``"true"``、``"false"``、``"unknown"``，避免客户端
在布尔值与字符串之间产生歧义。无法判断的文本属性统一使用 ``"unknown"``。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.analysis_v2 import (
    AnalysisStatus,
    AnalysisWarning,
    BodyObservability,
    ClothingItem,
    ConfidenceLevel,
    Evidence,
    FootState,
    PersonOrientation,
    SideBasis,
    SourceType,
    WearingRelation,
)

TriState = Literal["true", "false", "unknown"]


class SchemaBase(BaseModel):
    model_config = ConfigDict(extra="ignore")


class BoundingBox(SchemaBase):
    """归一化坐标（0~1），相对于图片左上角。"""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)


class Position(SchemaBase):
    description: str = "unknown"
    bbox: BoundingBox | None = None


class Visibility(SchemaBase):
    full_body_visible: TriState = "unknown"
    head_visible: TriState = "unknown"
    upper_body_visible: TriState = "unknown"
    legs_visible: TriState = "unknown"
    feet_visible: TriState = "unknown"


class HeadAttributes(SchemaBase):
    wearing_hat: TriState = "unknown"
    hat_type: str = "unknown"
    hat_color: str = "unknown"
    wearing_glasses: TriState = "unknown"


class UpperBodyAttributes(SchemaBase):
    wearing_top: TriState = "unknown"
    top_type: str = "unknown"
    top_color: str = "unknown"
    wearing_jacket: TriState = "unknown"
    jacket_color: str = "unknown"


class LowerBodyAttributes(SchemaBase):
    wearing_pants: TriState = "unknown"
    pants_type: str = "unknown"
    pants_color: str = "unknown"
    wearing_skirt: TriState = "unknown"
    skirt_color: str = "unknown"


class LegAttributes(SchemaBase):
    skin_visible: TriState = "unknown"
    wearing_tights: TriState = "unknown"
    wearing_stockings: TriState = "unknown"
    wearing_long_pants: TriState = "unknown"
    wearing_shorts: TriState = "unknown"
    leg_covering: str = "unknown"


class FeetAttributes(SchemaBase):
    wearing_socks: TriState = "unknown"
    socks_type: str = "unknown"
    socks_color: str = "unknown"
    wearing_shoes: TriState = "unknown"
    shoes_type: str = "unknown"
    shoes_color: str = "unknown"
    barefoot: TriState = "unknown"


class PersonConfidence(SchemaBase):
    head: float | None = Field(default=None, ge=0, le=1)
    upper_body: float | None = Field(default=None, ge=0, le=1)
    lower_body: float | None = Field(default=None, ge=0, le=1)
    legs: float | None = Field(default=None, ge=0, le=1)
    feet: float | None = Field(default=None, ge=0, le=1)


class PersonAnalysis(SchemaBase):
    person_id: int = Field(ge=1)
    position: Position = Field(default_factory=Position)
    visibility: Visibility = Field(default_factory=Visibility)
    head: HeadAttributes = Field(default_factory=HeadAttributes)
    upper_body: UpperBodyAttributes = Field(default_factory=UpperBodyAttributes)
    lower_body: LowerBodyAttributes = Field(default_factory=LowerBodyAttributes)
    legs: LegAttributes = Field(default_factory=LegAttributes)
    feet: FeetAttributes = Field(default_factory=FeetAttributes)
    confidence: PersonConfidence = Field(default_factory=PersonConfidence)
    evidence: dict[str, str] = Field(default_factory=dict)


class ImageInfo(SchemaBase):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    person_count: int = Field(ge=0)


class AnalysisData(SchemaBase):
    """图片分析数据。

    ``image_info`` / ``persons`` 为旧接口兼容字段；V2 字段采用追加方式，
    不影响旧客户端。
    """

    image_info: ImageInfo
    persons: list[PersonAnalysis]

    schema_version: int = 2
    analysis_id: str | None = None
    person_count: int | None = None
    source_type: SourceType | None = None
    status: AnalysisStatus | None = None
    person_orientation: PersonOrientation | None = None
    side_basis: SideBasis | None = None
    side_assignment_confidence: ConfidenceLevel | None = None
    items: list[ClothingItem] = Field(default_factory=list)
    body_observability: BodyObservability | None = None
    feet: list[FootState] = Field(default_factory=list)
    wearing_relations: list[WearingRelation] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    warnings: list[AnalysisWarning] = Field(default_factory=list)
    created_at: datetime | None = None
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    model: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)


class AnalyzeResponse(SchemaBase):
    success: Literal[True] = True
    message: str = "分析成功"
    data: AnalysisData


class ErrorDetail(SchemaBase):
    code: str
    detail: str
    details: list[dict[str, Any]] | None = None


class ErrorResponse(SchemaBase):
    success: Literal[False] = False
    message: str
    error: ErrorDetail
    request_id: str | None = None
