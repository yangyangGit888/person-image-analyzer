"""V2 统一分析数据结构。

图片和视频共用同一套核心结构：

- ``AnalysisResultV2``：一次分析的最终结果。
- ``ClothingItem``：下装、丝袜、普通袜子、鞋子的统一物品模型。
- ``Geometry``：物品级 Bounding Box 与坐标空间。
- ``BodyObservability``：身体各区域的可观察性和遮挡原因。
- ``WearingRelation``：左右脚/腿的穿戴关系。

旧的 ``app.schemas.analysis`` 保持不动，通过兼容适配器生成旧 ``persons``
响应，避免破坏已有客户端。
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PANTS_TYPES = frozenset(
    {
        "yoga_pants",
        "leggings",
        "jeans",
        "casual_pants",
        "suit_pants",
        "wide_leg_pants",
        "cargo_pants",
        "shorts",
        "hot_pants",
        "other",
        "unknown",
    }
)

SKIRT_TYPES = frozenset(
    {
        "pleated_skirt",
        "jk_skirt",
        "a_line_skirt",
        "pencil_skirt",
        "mini_skirt",
        "midi_skirt",
        "maxi_skirt",
        "denim_skirt",
        "puffy_skirt",
        "lolita_skirt",
        "other",
        "unknown",
    }
)

HOSIERY_TYPES = frozenset(
    {
        "pantyhose",
        "thigh_high",
        "knee_high",
        "calf_high",
        "ankle_high",
        "other",
        "unknown",
    }
)

SOCKS_TYPES = frozenset(
    {
        "no_show",
        "ankle",
        "crew",
        "knee_high",
        "thigh_high",
        "other",
        "unknown",
    }
)

SHOES_TYPES = frozenset(
    {
        "sneakers",
        "heels",
        "flats",
        "loafers",
        "boots",
        "sandals",
        "slippers",
        "other",
        "unknown",
    }
)

_TYPE_SETS: dict[str, frozenset[str]] = {
    "pants": PANTS_TYPES,
    "skirt": SKIRT_TYPES,
    "hosiery": HOSIERY_TYPES,
    "socks": SOCKS_TYPES,
    "shoes": SHOES_TYPES,
}


class SchemaBaseV2(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SourceType(StrEnum):
    IMAGE = "image"
    VIDEO = "video"


class AnalysisStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    UNSUPPORTED_MULTI_PERSON = "unsupported_multi_person"
    FAILED = "failed"


class ItemCategory(StrEnum):
    PANTS = "pants"
    SKIRT = "skirt"
    ONE_PIECE = "one_piece"
    HOSIERY = "hosiery"
    SOCKS = "socks"
    SHOES = "shoes"
    UNKNOWN = "unknown"


class PantsType(StrEnum):
    YOGA_PANTS = "yoga_pants"
    LEGGINGS = "leggings"
    JEANS = "jeans"
    CASUAL_PANTS = "casual_pants"
    SUIT_PANTS = "suit_pants"
    WIDE_LEG_PANTS = "wide_leg_pants"
    CARGO_PANTS = "cargo_pants"
    SHORTS = "shorts"
    HOT_PANTS = "hot_pants"
    OTHER = "other"
    UNKNOWN = "unknown"


class SkirtType(StrEnum):
    PLEATED_SKIRT = "pleated_skirt"
    JK_SKIRT = "jk_skirt"
    A_LINE_SKIRT = "a_line_skirt"
    PENCIL_SKIRT = "pencil_skirt"
    MINI_SKIRT = "mini_skirt"
    MIDI_SKIRT = "midi_skirt"
    MAXI_SKIRT = "maxi_skirt"
    DENIM_SKIRT = "denim_skirt"
    PUFFY_SKIRT = "puffy_skirt"
    LOLITA_SKIRT = "lolita_skirt"
    OTHER = "other"
    UNKNOWN = "unknown"


class HosieryType(StrEnum):
    PANTYHOSE = "pantyhose"
    THIGH_HIGH = "thigh_high"
    KNEE_HIGH = "knee_high"
    CALF_HIGH = "calf_high"
    ANKLE_HIGH = "ankle_high"
    OTHER = "other"
    UNKNOWN = "unknown"


class SocksType(StrEnum):
    NO_SHOW = "no_show"
    ANKLE = "ankle"
    CREW = "crew"
    KNEE_HIGH = "knee_high"
    THIGH_HIGH = "thigh_high"
    OTHER = "other"
    UNKNOWN = "unknown"


class ShoesType(StrEnum):
    SNEAKERS = "sneakers"
    HEELS = "heels"
    FLATS = "flats"
    LOAFERS = "loafers"
    BOOTS = "boots"
    SANDALS = "sandals"
    SLIPPERS = "slippers"
    OTHER = "other"
    UNKNOWN = "unknown"


class Presence(StrEnum):
    DETECTED = "detected"
    NOT_DETECTED = "not_detected"
    UNCERTAIN = "uncertain"
    NOT_VISIBLE = "not_visible"


class Visibility(StrEnum):
    CLEAR = "clear"
    PARTIAL = "partial"
    OCCLUDED = "occluded"
    NOT_VISIBLE = "not_visible"


class Side(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SideBasis(StrEnum):
    PERSON_OWN = "person_own"
    IMAGE_SIDE = "image_side"
    UNKNOWN = "unknown"


class PersonOrientation(StrEnum):
    FRONT = "front"
    BACK = "back"
    LEFT_PROFILE = "left_profile"
    RIGHT_PROFILE = "right_profile"
    THREE_QUARTER = "three_quarter"
    UNKNOWN = "unknown"


class OcclusionReason(StrEnum):
    LONG_SKIRT = "long_skirt"
    LONG_PANTS = "long_pants"
    OTHER_CLOTHING = "other_clothing"
    OBJECT_OCCLUSION = "object_occlusion"
    BODY_POSE = "body_pose"
    OUT_OF_FRAME = "out_of_frame"
    LOW_RESOLUTION = "low_resolution"
    POOR_LIGHTING = "poor_lighting"
    UNKNOWN = "unknown"


class BarefootStatus(StrEnum):
    BAREFOOT = "barefoot"
    NOT_BAREFOOT = "not_barefoot"
    UNKNOWN = "unknown"
    NOT_VISIBLE = "not_visible"


class MaskStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    READY = "ready"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class GeometryPrecision(StrEnum):
    ESTIMATE = "estimate"
    COARSE = "coarse"
    EXACT = "exact"


class EvidenceSource(StrEnum):
    OVERVIEW = "overview"
    CROP = "crop"
    FRAME = "frame"
    RULE = "rule"


class BodyRegion(StrEnum):
    LOWER_BODY = "lower_body"
    LEFT_LEG = "left_leg"
    RIGHT_LEG = "right_leg"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"


class RelationType(StrEnum):
    HOSIERY_COVERS_FEET = "hosiery_covers_feet"
    REGULAR_SOCKS_OVER_HOSIERY = "regular_socks_over_hosiery"
    SHOES_OVER_SOCKS = "shoes_over_socks"
    SHOES_OVER_HOSIERY = "shoes_over_hosiery"


class RelationStatus(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class ItemAttributes(SchemaBaseV2):
    """物品属性的开放集合。

    不同 category 使用不同字段，未使用的字段保持 ``None``。允许额外字段，
    以便未来增加眼镜、耳环、发型等物品时不需要破坏当前结构。
    """

    model_config = ConfigDict(extra="allow")

    color: str | None = None
    length: str | None = None
    pattern: str | None = None
    visual_material: str | None = None
    texture: str | None = None
    visual_thickness: str | None = None
    sheerness_level: int | None = Field(default=None, ge=0, le=100)
    covers_feet: bool | None = None
    fit: str | None = None
    shoe_shape: str | None = None
    decorations: list[str] = Field(default_factory=list)
    notes: str | None = None


class Geometry(SchemaBaseV2):
    """物品定位信息。

    ``bbox`` 固定为 ``[x_min, y_min, x_max, y_max]``，坐标单位为原始图片或
    证据帧像素。视频物品通过 ``frame_id`` 和 ``timestamp`` 关联证据帧。
    """

    coordinate_space: Literal["source_pixels"] = "source_pixels"
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    bbox: list[int] | None = Field(default=None, min_length=4, max_length=4)
    precision: GeometryPrecision = GeometryPrecision.ESTIMATE
    method: str = "model_estimate"
    frame_id: str | None = None
    timestamp: float | None = Field(default=None, ge=0)

    @field_validator("bbox")
    @classmethod
    def _validate_bbox(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("bbox 必须包含 4 个整数")
        x_min, y_min, x_max, y_max = value
        if x_min < 0 or y_min < 0:
            raise ValueError("bbox 左上角不能为负数")
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("bbox 必须满足 x_max > x_min 且 y_max > y_min")
        return value

    @model_validator(mode="after")
    def _bbox_within_source(self) -> Geometry:
        if self.bbox is None:
            return self
        x_min, y_min, x_max, y_max = self.bbox
        if x_max > self.source_width or y_max > self.source_height:
            raise ValueError("bbox 超出原始图片或证据帧边界")
        return self


class Evidence(SchemaBaseV2):
    frame_id: str | None = None
    timestamp: float | None = Field(default=None, ge=0)
    bbox: Geometry | None = None
    observation: str = ""
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    source: EvidenceSource = EvidenceSource.OVERVIEW


class ClothingItem(SchemaBaseV2):
    """统一服饰物品。

    下装使用 ``category=pants/skirt/one_piece``；丝袜、普通袜子、鞋子分别使用
    ``hosiery/socks/shoes``。左右腿/脚物品必须填写 ``side``。
    """

    item_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    analysis_id: str | None = None
    source_id: str | None = None
    category: ItemCategory = ItemCategory.UNKNOWN
    type: str = "unknown"
    side: Side = Side.UNKNOWN
    image_side: Side = Side.UNKNOWN
    presence: Presence = Presence.UNCERTAIN
    visibility: Visibility = Visibility.NOT_VISIBLE
    attributes: ItemAttributes = Field(default_factory=ItemAttributes)
    style_tags: list[str] = Field(default_factory=list)
    design_features: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    geometry: Geometry | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    occlusion_reason: list[OcclusionReason] = Field(default_factory=list)
    mask_status: MaskStatus = MaskStatus.NOT_REQUESTED
    mask_reference: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("type")
    @classmethod
    def _normalize_type(cls, value: str) -> str:
        return value.strip().lower() if value and value.strip() else "unknown"

    @field_validator("style_tags", "design_features")
    @classmethod
    def _normalize_tags(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        for item in value:
            text = str(item).strip().lower().replace(" ", "_")
            if text and text not in result:
                result.append(text)
        return result

    def ensure_type_allowed(self) -> None:
        """把不属于当前 category 的 type 安全降级为 unknown。"""

        allowed = _TYPE_SETS.get(self.category.value)
        if allowed is not None and self.type not in allowed:
            self.type = "unknown"


class RegionObservability(SchemaBaseV2):
    region: BodyRegion
    visibility: Visibility = Visibility.NOT_VISIBLE
    occlusion_reason: list[OcclusionReason] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    evidence: list[Evidence] = Field(default_factory=list)
    notes: str | None = None


def _default_region(region: BodyRegion) -> RegionObservability:
    return RegionObservability(region=region)


class BodyObservability(SchemaBaseV2):
    lower_body: RegionObservability = Field(
        default_factory=lambda: _default_region(BodyRegion.LOWER_BODY)
    )
    left_leg: RegionObservability = Field(
        default_factory=lambda: _default_region(BodyRegion.LEFT_LEG)
    )
    right_leg: RegionObservability = Field(
        default_factory=lambda: _default_region(BodyRegion.RIGHT_LEG)
    )
    left_foot: RegionObservability = Field(
        default_factory=lambda: _default_region(BodyRegion.LEFT_FOOT)
    )
    right_foot: RegionObservability = Field(
        default_factory=lambda: _default_region(BodyRegion.RIGHT_FOOT)
    )


class FootState(SchemaBaseV2):
    side: Side
    image_side: Side = Side.UNKNOWN
    barefoot: BarefootStatus = BarefootStatus.UNKNOWN
    visibility: Visibility = Visibility.NOT_VISIBLE
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    item_ids: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    uncertainty_reason: str | None = None


class WearingRelation(SchemaBaseV2):
    relation_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    type: RelationType
    side: Side
    status: RelationStatus = RelationStatus.UNKNOWN
    evidence: list[Evidence] = Field(default_factory=list)
    conflict: bool = False
    notes: str | None = None


class AnalysisWarning(SchemaBaseV2):
    code: str
    message: str
    detail: str | None = None


class AnalysisResultV2(SchemaBaseV2):
    schema_version: int = 2
    analysis_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    source_id: str | None = None
    source_type: SourceType = SourceType.IMAGE
    status: AnalysisStatus = AnalysisStatus.COMPLETED
    person_count: int = Field(default=0, ge=0)
    person_orientation: PersonOrientation = PersonOrientation.UNKNOWN
    side_basis: SideBasis = SideBasis.UNKNOWN
    side_assignment_confidence: ConfidenceLevel = ConfidenceLevel.LOW
    items: list[ClothingItem] = Field(default_factory=list)
    body_observability: BodyObservability = Field(default_factory=BodyObservability)
    feet: list[FootState] = Field(default_factory=list)
    wearing_relations: list[WearingRelation] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    warnings: list[AnalysisWarning] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    model: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)

    def normalized_items(self) -> list[ClothingItem]:
        """返回 type 已按 category 校验的物品列表。"""

        for item in self.items:
            item.ensure_type_allowed()
        return self.items


class AnalysisV2Data(SchemaBaseV2):
    """API 追加字段的数据容器，便于未来逐步切换主响应。"""

    analysis: AnalysisResultV2
    legacy_persons: list[dict[str, Any]] = Field(default_factory=list)
