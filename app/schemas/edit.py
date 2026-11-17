"""人像局部编辑的数据模型与 API 契约。

内部记录（*Record）会包含本地存储路径，API 只返回 *View，避免泄露服务器路径。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EditSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")


class StrictEditSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegionType(str, Enum):
    HAT = "hat"
    GLASSES = "glasses"
    HAIR = "hair"
    EARRING = "earring"
    TOP = "top"
    PANTS = "pants"
    SHOES = "shoes"


class EditOperation(str, Enum):
    ADD = "add"
    REPLACE = "replace"
    REMOVE = "remove"
    MODIFY = "modify"


SUPPORTED_OPERATIONS = {EditOperation.ADD, EditOperation.REPLACE}


class ExistenceStatus(str, Enum):
    EXISTING = "existing"
    MISSING = "missing"
    UNKNOWN = "unknown"


class MaskStrategy(str, Enum):
    OBJECT = "object"
    PLACEMENT = "placement"


class TaskStatus(str, Enum):
    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GenerationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class PixelBox(EditSchema):
    """以规范化图片的左上角为原点的像素坐标框。"""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.right, self.bottom


class MaskQuality(EditSchema):
    score: float | None = Field(default=None, ge=0, le=1)
    area_ratio: float = Field(default=0, ge=0, le=1)
    valid: bool = True
    reason: str | None = None


class StoredImage(EditSchema):
    """内部图片记录，包含本地相对路径，禁止直接返回给客户端。"""

    image_id: str
    original_width: int = Field(gt=0)
    original_height: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: str
    file_size: int = Field(ge=0)
    source_path: str
    canonical_path: str
    person_count: int = Field(default=0, ge=0)
    analysis_status: AnalysisStatus = AnalysisStatus.PENDING
    created_at: datetime
    expires_at: datetime


class ImageView(EditSchema):
    image_id: str
    original_width: int
    original_height: int
    width: int
    height: int
    format: str
    file_size: int
    person_count: int
    analysis_status: AnalysisStatus
    created_at: datetime
    expires_at: datetime

    @classmethod
    def from_record(cls, record: StoredImage) -> ImageView:
        return cls(
            image_id=record.image_id,
            original_width=record.original_width,
            original_height=record.original_height,
            width=record.width,
            height=record.height,
            format=record.format,
            file_size=record.file_size,
            person_count=record.person_count,
            analysis_status=record.analysis_status,
            created_at=record.created_at,
            expires_at=record.expires_at,
        )


class PersonRecord(EditSchema):
    person_id: str
    image_id: str
    ordinal: int = Field(ge=1)
    bbox: PixelBox
    person_mask_path: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    created_at: datetime


class PersonView(EditSchema):
    person_id: str
    image_id: str
    ordinal: int
    bbox: PixelBox
    mask_ready: bool
    confidence: float | None
    created_at: datetime

    @classmethod
    def from_record(cls, record: PersonRecord) -> PersonView:
        return cls(
            person_id=record.person_id,
            image_id=record.image_id,
            ordinal=record.ordinal,
            bbox=record.bbox,
            mask_ready=bool(record.person_mask_path),
            confidence=record.confidence,
            created_at=record.created_at,
        )


class RegionRecord(EditSchema):
    region_id: str
    image_id: str
    person_id: str
    region_type: RegionType
    bbox: PixelBox | None = None
    mask_path: str | None = None
    mask_strategy: MaskStrategy | None = None
    detection_method: str = "unknown"
    segmentation_method: str = "unknown"
    existence_status: ExistenceStatus = ExistenceStatus.UNKNOWN
    confidence: float | None = Field(default=None, ge=0, le=1)
    mask_quality: MaskQuality | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
    created_at: datetime


class RegionView(EditSchema):
    region_id: str
    image_id: str
    person_id: str
    region_type: RegionType
    bbox: PixelBox | None
    mask_ready: bool
    mask_strategy: MaskStrategy | None
    detection_method: str
    segmentation_method: str
    existence_status: ExistenceStatus
    confidence: float | None
    mask_quality: MaskQuality | None
    attributes: dict[str, str]
    created_at: datetime

    @classmethod
    def from_record(cls, record: RegionRecord) -> RegionView:
        return cls(
            region_id=record.region_id,
            image_id=record.image_id,
            person_id=record.person_id,
            region_type=record.region_type,
            bbox=record.bbox,
            mask_ready=bool(record.mask_path),
            mask_strategy=record.mask_strategy,
            detection_method=record.detection_method,
            segmentation_method=record.segmentation_method,
            existence_status=record.existence_status,
            confidence=record.confidence,
            mask_quality=record.mask_quality,
            attributes=record.attributes,
            created_at=record.created_at,
        )


class EditTaskRecord(EditSchema):
    task_id: str
    image_id: str
    person_id: str
    region_id: str
    region_type: RegionType
    operation: EditOperation
    selected_options: dict[str, str]
    prompt: str
    mask_path: str | None = None
    task_status: TaskStatus = TaskStatus.DRAFT
    provider_task_id: str | None = None
    result_path: str | None = None
    error_code: str | None = None
    error_detail: str | None = None
    created_at: datetime
    updated_at: datetime


class EditTaskView(EditSchema):
    task_id: str
    image_id: str
    person_id: str
    region_id: str
    region_type: RegionType
    operation: EditOperation
    selected_options: dict[str, str]
    prompt: str
    mask_ready: bool
    task_status: TaskStatus
    provider_task_id: str | None
    result_path_ready: bool
    error_code: str | None
    error_detail: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, record: EditTaskRecord) -> EditTaskView:
        return cls(
            task_id=record.task_id,
            image_id=record.image_id,
            person_id=record.person_id,
            region_id=record.region_id,
            region_type=record.region_type,
            operation=record.operation,
            selected_options=record.selected_options,
            prompt=record.prompt,
            mask_ready=bool(record.mask_path),
            task_status=record.task_status,
            provider_task_id=record.provider_task_id,
            result_path_ready=bool(record.result_path),
            error_code=record.error_code,
            error_detail=record.error_detail,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


class GenerationStepRecord(EditSchema):
    generation_id: str
    sequence: int = Field(ge=1)
    task_id: str
    region_type: RegionType
    provider_task_id: str | None = None
    status: StepStatus = StepStatus.PENDING
    input_path: str | None = None
    output_path: str | None = None
    error_code: str | None = None
    error_detail: str | None = None


class GenerationStepView(EditSchema):
    sequence: int
    task_id: str
    region_type: RegionType
    provider_task_id: str | None
    status: StepStatus
    output_ready: bool
    error_code: str | None
    error_detail: str | None

    @classmethod
    def from_record(cls, record: GenerationStepRecord) -> GenerationStepView:
        return cls(
            sequence=record.sequence,
            task_id=record.task_id,
            region_type=record.region_type,
            provider_task_id=record.provider_task_id,
            status=record.status,
            output_ready=bool(record.output_path),
            error_code=record.error_code,
            error_detail=record.error_detail,
        )


class GenerationRecord(EditSchema):
    generation_id: str
    image_id: str
    person_id: str
    task_ids: list[str]
    plan_hash: str
    status: GenerationStatus = GenerationStatus.PENDING
    current_step: int = Field(default=0, ge=0)
    result_path: str | None = None
    error_code: str | None = None
    error_detail: str | None = None
    idempotency_key: str | None = None
    created_at: datetime
    updated_at: datetime


class GenerationView(EditSchema):
    generation_id: str
    image_id: str
    person_id: str
    task_ids: list[str]
    status: GenerationStatus
    current_step: int
    result_ready: bool
    error_code: str | None
    error_detail: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, record: GenerationRecord) -> GenerationView:
        return cls(
            generation_id=record.generation_id,
            image_id=record.image_id,
            person_id=record.person_id,
            task_ids=record.task_ids,
            status=record.status,
            current_step=record.current_step,
            result_ready=bool(record.result_path),
            error_code=record.error_code,
            error_detail=record.error_detail,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


class ImageUploadResponse(EditSchema):
    success: Literal[True] = True
    message: str = "上传成功"
    data: ImageView


class EditAnalyzeData(EditSchema):
    image: ImageView
    persons: list[PersonView]
    regions: list[RegionView]


class EditAnalyzeResponse(EditSchema):
    success: Literal[True] = True
    message: str = "分析成功"
    data: EditAnalyzeData


class PersonListResponse(EditSchema):
    success: Literal[True] = True
    message: str = "ok"
    data: list[PersonView]


class RegionListResponse(EditSchema):
    success: Literal[True] = True
    message: str = "ok"
    data: list[RegionView]


class CreateEditTaskRequest(StrictEditSchema):
    image_id: str
    person_id: str
    region_type: RegionType
    operation: EditOperation
    style_id: str
    color_id: str

    @model_validator(mode="after")
    def _check_operation(self) -> CreateEditTaskRequest:
        if self.operation not in SUPPORTED_OPERATIONS:
            raise ValueError("当前版本仅支持 add 和 replace 操作")
        return self


class UpdateEditTaskRequest(StrictEditSchema):
    operation: EditOperation | None = None
    style_id: str | None = None
    color_id: str | None = None

    @model_validator(mode="after")
    def _check_operation(self) -> UpdateEditTaskRequest:
        if self.operation is not None and self.operation not in SUPPORTED_OPERATIONS:
            raise ValueError("当前版本仅支持 add 和 replace 操作")
        if not any(
            value is not None
            for value in (self.operation, self.style_id, self.color_id)
        ):
            raise ValueError("至少提供一个待更新字段")
        return self


class EditTaskResponse(EditSchema):
    success: Literal[True] = True
    message: str = "ok"
    data: EditTaskView


class EditTaskListResponse(EditSchema):
    success: Literal[True] = True
    message: str = "ok"
    data: list[EditTaskView]


class CreateGenerationRequest(StrictEditSchema):
    image_id: str
    person_id: str
    task_ids: list[str] = Field(default_factory=list)
    idempotency_key: str | None = Field(default=None, max_length=128)


class GenerationData(EditSchema):
    generation: GenerationView
    steps: list[GenerationStepView]


class GenerationResponse(EditSchema):
    success: Literal[True] = True
    message: str = "ok"
    data: GenerationData


class EditStatusResponse(EditSchema):
    success: Literal[True] = True
    message: str = "ok"
    data: EditTaskView
