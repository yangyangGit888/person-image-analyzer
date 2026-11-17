"""编辑专用人物分析编排：DeepSeek 人物框 + SAM2 + 区域检测 + Mask。"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import numpy as np

from app.ai.base import ImageInput
from app.core.config import Settings
from app.core.exceptions import (
    EditValidationError,
    RegionDetectionError,
    SegmentationError,
)
from app.repositories.base import EditRepository
from app.schemas.analysis import PersonAnalysis
from app.schemas.edit import (
    AnalysisStatus,
    ExistenceStatus,
    MaskStrategy,
    PersonRecord,
    PixelBox,
    RegionRecord,
    RegionType,
    StoredImage,
)
from app.services.analysis_service import AnalysisService
from app.services.edit_image_service import EditImageService
from app.services.mask_service import MaskService
from app.services.region_detection_service import DetectedRegion, RegionDetectionService
from app.services.segmentation_service import SegmentationService
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)


class EditAnalysisService:
    def __init__(
        self,
        settings: Settings,
        repository: EditRepository,
        storage: StorageService,
        image_service: EditImageService,
        analysis_service: AnalysisService,
        region_detection_service: RegionDetectionService,
        segmentation_service: SegmentationService,
        mask_service: MaskService,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._storage = storage
        self._image_service = image_service
        self._analysis_service = analysis_service
        self._region_detection_service = region_detection_service
        self._segmentation_service = segmentation_service
        self._mask_service = mask_service

    async def analyze(
        self,
        image_id: str,
    ) -> tuple[StoredImage, list[PersonRecord], list[RegionRecord]]:
        image = await self._image_service.get_image(image_id)
        if image.analysis_status is AnalysisStatus.COMPLETED:
            persons = await asyncio.to_thread(self._repository.list_persons, image_id)
            regions = await asyncio.to_thread(
                self._repository.list_regions_by_image, image_id
            )
            return image, persons, regions

        try:
            return await self._analyze_new(image)
        except Exception:
            image.analysis_status = AnalysisStatus.FAILED
            await asyncio.to_thread(self._repository.update_image, image)
            raise

    async def _analyze_new(
        self,
        image: StoredImage,
    ) -> tuple[StoredImage, list[PersonRecord], list[RegionRecord]]:
        image_bytes = await self._image_service.read_image_bytes(image)
        image_np = self._segmentation_service.image_from_bytes(image_bytes)
        image_input = ImageInput(
            data=image_bytes,
            mime_type="image/jpeg",
            width=image.width,
            height=image.height,
        )

        analysis = await self._analysis_service.analyze(image_input)
        if len(analysis.persons) > self._settings.edit_max_persons:
            raise EditValidationError(
                detail=f"图片中人物过多，最多支持 {self._settings.edit_max_persons} 人",
            )
        if len(analysis.persons) > 1 and any(
            person.position.bbox is None for person in analysis.persons
        ):
            raise RegionDetectionError(
                detail=(
                    "多人图片中有人员缺少定位框，无法保证人物分离，"
                    "请换用更高分辨率或人物更清晰的照片"
                )
            )

        person_boxes = [
            self._person_box(person, image)
            for person in analysis.persons
        ]
        person_masks = await self._segmentation_service.segment_persons(
            image_np, person_boxes
        )

        now = datetime.now(UTC)
        person_records: list[PersonRecord] = []
        person_mask_arrays: list[np.ndarray] = []
        for index, (_, box, segmented) in enumerate(
            zip(analysis.persons, person_boxes, person_masks, strict=True),
            start=1,
        ):
            mask = segmented.mask.astype(bool)
            if mask.shape != (image.height, image.width):
                raise SegmentationError(detail="人物 Mask 尺寸与图片不一致")
            area_ratio = float(mask.mean())
            if area_ratio < 0.002 or area_ratio > 0.98:
                raise SegmentationError(detail="人物 Mask 面积异常，无法继续编辑")
            person_id = self._storage.new_id()
            mask_path = self._storage.relative_path(
                "images", image.image_id, "persons", f"{person_id}.png"
            )
            await self._mask_service.save_mask(mask_path, mask)
            person_records.append(
                PersonRecord(
                    person_id=person_id,
                    image_id=image.image_id,
                    ordinal=index,
                    bbox=box,
                    person_mask_path=mask_path,
                    confidence=segmented.score,
                    created_at=now,
                )
            )
            person_mask_arrays.append(mask)

        detected_regions = await self._region_detection_service.detect(
            image_input,
            analysis.persons,
            person_boxes,
        )

        region_records: list[RegionRecord] = []
        for person_index, (person_record, detected_list) in enumerate(
            zip(person_records, detected_regions, strict=True)
        ):
            own_mask = person_mask_arrays[person_index]
            other_mask = self._other_person_mask(person_mask_arrays, person_index)
            for detected in detected_list:
                region_record = await self._build_region(
                    image=image,
                    image_np=image_np,
                    person=person_record,
                    detected=detected,
                    person_mask=own_mask,
                    other_person_mask=other_mask,
                    now=now,
                )
                region_records.append(region_record)

        for person in person_records:
            await asyncio.to_thread(self._repository.save_person, person)
        for region in region_records:
            await asyncio.to_thread(self._repository.save_region, region)

        image = await self._image_service.mark_analyzed(
            image, len(person_records)
        )
        logger.info(
            "编辑分析完成: image_id=%s persons=%s regions=%s",
            image.image_id,
            len(person_records),
            len(region_records),
        )
        return image, person_records, region_records

    async def _build_region(
        self,
        *,
        image: StoredImage,
        image_np: np.ndarray,
        person: PersonRecord,
        detected: DetectedRegion,
        person_mask: np.ndarray,
        other_person_mask: np.ndarray | None,
        now: datetime,
    ) -> RegionRecord:
        region_id = self._storage.new_id()
        mask_path = self._storage.relative_path(
            "images",
            image.image_id,
            "regions",
            f"{region_id}.png",
        )
        mask: np.ndarray | None = None
        strategy: MaskStrategy | None = None
        segmentation_method = "none"
        segment_score: float | None = None

        if (
            detected.existence_status is ExistenceStatus.EXISTING
            and detected.bbox is not None
        ):
            strategy = MaskStrategy.OBJECT
            try:
                segmented = await self._segmentation_service.segment_object(
                    image_np,
                    detected.bbox,
                    prompt_id=region_id,
                )
                mask = segmented.mask.astype(bool)
                segmentation_method = segmented.method
                segment_score = segmented.score
                if detected.region_type in {
                    RegionType.HAIR,
                    RegionType.TOP,
                    RegionType.PANTS,
                    RegionType.SHOES,
                }:
                    mask = self._mask_service.clip(mask, person_mask, expand=12)
                mask = self._mask_service.subtract(mask, other_person_mask)
            except SegmentationError:
                logger.warning(
                    "目标分割失败: image=%s person=%s region=%s",
                    image.image_id,
                    person.person_id,
                    detected.region_type.value,
                )
                mask = None
                strategy = None
        elif detected.existence_status in {
            ExistenceStatus.MISSING,
            ExistenceStatus.UNKNOWN,
        }:
            strategy = MaskStrategy.PLACEMENT
            segmentation_method = "geometric_placement"
            try:
                mask = self._mask_service.placement_mask(
                    region_type=detected.region_type,
                    person_bbox=person.bbox,
                    image_width=image.width,
                    image_height=image.height,
                    person_mask=person_mask,
                )
                mask = self._mask_service.subtract(mask, other_person_mask)
            except ValueError:
                mask = None

        quality = None
        if mask is not None:
            quality = self._mask_service.validate(
                mask,
                image_width=image.width,
                image_height=image.height,
                region_type=detected.region_type,
                segment_score=segment_score,
                strategy=strategy or MaskStrategy.PLACEMENT,
                person_mask=person_mask,
            )
            if (
                quality.valid
                and segment_score is not None
                and segment_score < self._settings.segmentation_min_score
            ):
                quality.reason = "分割置信度较低，建议人工确认结果"
            if quality.valid:
                await self._mask_service.save_mask(mask_path, mask)
            else:
                mask_path = ""
                logger.info(
                    "Mask 校验失败: region=%s reason=%s",
                    detected.region_type.value,
                    quality.reason,
                )
        else:
            mask_path = ""

        return RegionRecord(
            region_id=region_id,
            image_id=image.image_id,
            person_id=person.person_id,
            region_type=detected.region_type,
            bbox=detected.bbox,
            mask_path=mask_path or None,
            mask_strategy=strategy,
            detection_method=detected.detection_method,
            segmentation_method=segmentation_method,
            existence_status=detected.existence_status,
            confidence=detected.confidence,
            mask_quality=quality,
            attributes=detected.attributes,
            created_at=now,
        )

    @staticmethod
    def _other_person_mask(
        masks: list[np.ndarray],
        current_index: int,
    ) -> np.ndarray | None:
        if len(masks) <= 1:
            return None
        result = np.zeros_like(masks[0], dtype=bool)
        for index, mask in enumerate(masks):
            if index != current_index:
                result |= mask
        return result

    @staticmethod
    def _person_box(person: PersonAnalysis, image: StoredImage) -> PixelBox:
        bbox = person.position.bbox
        if bbox is None:
            return PixelBox(
                x=int(image.width * 0.02),
                y=int(image.height * 0.02),
                width=max(1, int(image.width * 0.96)),
                height=max(1, int(image.height * 0.96)),
            )
        pad_x = bbox.width * image.width * 0.02
        pad_y = bbox.height * image.height * 0.02
        x1 = max(0, round(bbox.x * image.width - pad_x))
        y1 = max(0, round(bbox.y * image.height - pad_y))
        x2 = min(
            image.width,
            round((bbox.x + bbox.width) * image.width + pad_x),
        )
        y2 = min(
            image.height,
            round((bbox.y + bbox.height) * image.height + pad_y),
        )
        if x2 <= x1 or y2 <= y1:
            raise RegionDetectionError(detail="人物 bbox 无效")
        return PixelBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
