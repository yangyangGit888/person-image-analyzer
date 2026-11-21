"""按需 Mask 的数据结构。

第一阶段只实现 BBox 矩形粗略 Mask，必须通过 ``precision=coarse`` 明确说明
它不是像素级精确分割。未来接入 SAM 等分割模型时复用同一接口。
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from app.schemas.analysis_v2 import Geometry, SchemaBaseV2


class MaskPrecision(StrEnum):
    COARSE = "coarse"
    PRECISE = "precise"


class MaskStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    READY = "ready"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class MaskRecord(SchemaBaseV2):
    """内部 Mask 记录，``path`` 不返回给客户端。"""

    mask_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    item_id: str
    analysis_id: str | None = None
    source_id: str | None = None
    frame_id: str | None = None
    path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    method: str = "bbox_rectangle"
    precision: MaskPrecision = MaskPrecision.COARSE
    status: MaskStatus = MaskStatus.READY
    geometry: Geometry | None = None
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None


class MaskView(SchemaBaseV2):
    mask_id: str
    item_id: str
    analysis_id: str | None = None
    source_id: str | None = None
    frame_id: str | None = None
    mask_reference: str
    width: int
    height: int
    method: str
    precision: MaskPrecision
    status: MaskStatus
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime
    expires_at: datetime | None = None

    @classmethod
    def from_record(cls, record: MaskRecord) -> MaskView:
        return cls(
            mask_id=record.mask_id,
            item_id=record.item_id,
            analysis_id=record.analysis_id,
            source_id=record.source_id,
            frame_id=record.frame_id,
            mask_reference=record.mask_id,
            width=record.width,
            height=record.height,
            method=record.method,
            precision=record.precision,
            status=record.status,
            warnings=record.warnings,
            created_at=record.created_at,
            expires_at=record.expires_at,
        )


class MaskResponse(SchemaBaseV2):
    success: Literal[True] = True
    message: str = "Mask 已生成"
    data: MaskView
