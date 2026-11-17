"""Mask 生成、校验与文件转换。

提交万相的 Mask 必须是纯 0/255 的二值 PNG；本地合成可以另行使用软 Mask。
"""

from __future__ import annotations

import asyncio
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app.catalogs.region_definitions import get_region_definition
from app.schemas.edit import MaskQuality, MaskStrategy, PixelBox, RegionType
from app.services.storage_service import StorageService


class MaskService:
    def __init__(self, storage: StorageService) -> None:
        self._storage = storage

    @staticmethod
    def empty_mask(width: int, height: int) -> np.ndarray:
        return np.zeros((height, width), dtype=bool)

    @staticmethod
    def binary(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if mask.dtype == bool:
            return mask
        return np.asarray(mask) > threshold

    @staticmethod
    def union(*masks: np.ndarray) -> np.ndarray:
        if not masks:
            return np.zeros((0, 0), dtype=bool)
        result = np.zeros_like(masks[0], dtype=bool)
        for mask in masks:
            if mask.shape != result.shape:
                raise ValueError("Mask 尺寸不一致")
            result |= mask
        return result

    @staticmethod
    def dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
        if pixels <= 0:
            return mask.astype(bool, copy=True)
        size = pixels * 2 + 1
        image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        image = image.filter(ImageFilter.MaxFilter(size))
        return np.asarray(image) > 127

    @staticmethod
    def erode(mask: np.ndarray, pixels: int) -> np.ndarray:
        if pixels <= 0:
            return mask.astype(bool, copy=True)
        size = pixels * 2 + 1
        image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        image = image.filter(ImageFilter.MinFilter(size))
        return np.asarray(image) > 127

    @staticmethod
    def clip(mask: np.ndarray, container: np.ndarray | None, expand: int = 0) -> np.ndarray:
        if container is None:
            return mask
        allowed = MaskService.dilate(container.astype(bool), expand)
        return mask.astype(bool) & allowed

    @staticmethod
    def subtract(mask: np.ndarray, other: np.ndarray | None) -> np.ndarray:
        if other is None:
            return mask
        return mask.astype(bool) & ~other.astype(bool)

    @staticmethod
    def _clip_box(
        box: PixelBox, width: int, height: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(width - 1, box.x))
        y1 = max(0, min(height - 1, box.y))
        x2 = max(x1 + 1, min(width, box.right))
        y2 = max(y1 + 1, min(height, box.bottom))
        return x1, y1, x2, y2

    def validate(
        self,
        mask: np.ndarray,
        *,
        image_width: int,
        image_height: int,
        region_type: RegionType,
        segment_score: float | None = None,
        strategy: MaskStrategy = MaskStrategy.OBJECT,
        person_mask: np.ndarray | None = None,
    ) -> MaskQuality:
        definition = get_region_definition(region_type)
        if mask.ndim != 2 or mask.shape != (image_height, image_width):
            return MaskQuality(
                score=segment_score,
                valid=False,
                reason="Mask 尺寸与图片不一致",
            )
        binary = mask.astype(bool)
        area = int(np.count_nonzero(binary))
        area_ratio = area / float(image_width * image_height)
        if area == 0:
            return MaskQuality(score=segment_score, area_ratio=0, valid=False, reason="Mask 为空")
        if strategy is MaskStrategy.OBJECT:
            min_ratio = definition.object_mask_min_area_ratio
            max_ratio = definition.object_mask_max_area_ratio
        else:
            min_ratio = definition.placement_mask_min_area_ratio
            max_ratio = definition.placement_mask_max_area_ratio
        if area_ratio < min_ratio:
            return MaskQuality(
                score=segment_score,
                area_ratio=area_ratio,
                valid=False,
                reason="Mask 面积过小",
            )
        if area_ratio > max_ratio:
            return MaskQuality(
                score=segment_score,
                area_ratio=area_ratio,
                valid=False,
                reason="Mask 面积过大，可能覆盖了非目标区域",
            )
        if person_mask is not None and region_type in {
            RegionType.TOP,
            RegionType.PANTS,
            RegionType.SHOES,
        }:
            overlap = int(np.count_nonzero(binary & person_mask.astype(bool)))
            if overlap / max(1, area) < 0.25:
                return MaskQuality(
                    score=segment_score,
                    area_ratio=area_ratio,
                    valid=False,
                    reason="Mask 与人物区域重叠过低",
                )
        return MaskQuality(
            score=segment_score,
            area_ratio=area_ratio,
            valid=True,
        )

    async def save_mask(self, relative_path: str, mask: np.ndarray) -> str:
        data = await asyncio.to_thread(self.mask_to_png_bytes, mask)
        return await self._storage.save_bytes(relative_path, data)

    @staticmethod
    def mask_to_png_bytes(mask: np.ndarray) -> bytes:
        binary = mask.astype(bool).astype(np.uint8) * 255
        image = Image.fromarray(binary, mode="L")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    async def load_mask(self, relative_path: str) -> np.ndarray:
        data = await self._storage.read_bytes(relative_path)
        return await asyncio.to_thread(self.png_bytes_to_mask, data)

    @staticmethod
    def png_bytes_to_mask(data: bytes) -> np.ndarray:
        with Image.open(BytesIO(data)) as image:
            return np.asarray(image.convert("L")) > 127

    # ---------- Placement Mask ----------
    @staticmethod
    def _head_geometry(
        person_mask: np.ndarray | None,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> tuple[int, int, int, int]:
        """估计头顶、头部左右边界和头部高度。

        有 SAM2 人物 Mask 时，使用人物最上方 15% 区域估计头部宽度，
        比直接按整个身体 bbox 取 30% 更不容易把帽子放到眼睛上。
        """

        person_width = max(1, x2 - x1)
        person_height = max(1, y2 - y1)
        if person_mask is not None and person_mask.any():
            rows = np.where(person_mask.any(axis=1))[0]
            if rows.size:
                top = int(rows.min())
                bottom = int(rows.max())
                sample_bottom = min(
                    bottom,
                    top + max(20, int((bottom - top) * 0.15)),
                )
                sample = person_mask[top : sample_bottom + 1]
                cols = np.where(sample.any(axis=0))[0]
                if cols.size:
                    left = int(cols.min())
                    right = int(cols.max())
                    head_w = max(20, right - left)
                    head_h = max(
                        int(head_w * 0.9),
                        min(int((bottom - top) * 0.2), int(head_w * 1.25)),
                    )
                    return left, top, right, head_h

        head_w = max(20, int(person_width * 0.45))
        head_x1 = x1 + (person_width - head_w) // 2
        head_h = max(
            int(head_w * 0.9),
            min(int(person_height * 0.18), int(head_w * 1.25)),
        )
        return head_x1, y1, head_x1 + head_w, head_h

    def placement_mask(
        self,
        *,
        region_type: RegionType,
        person_bbox: PixelBox,
        image_width: int,
        image_height: int,
        person_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        canvas = Image.new("L", (image_width, image_height), 0)
        draw = ImageDraw.Draw(canvas)
        x1, y1, x2, y2 = self._clip_box(person_bbox, image_width, image_height)
        width, height = x2 - x1, y2 - y1

        head_x1, head_y1, head_x2, head_h = self._head_geometry(
            person_mask,
            x1,
            y1,
            x2,
            y2,
        )
        head_y2 = head_y1 + head_h
        head_w = max(4, head_x2 - head_x1)

        def ellipse(box: tuple[float, float, float, float]) -> None:
            draw.ellipse(
                (
                    max(0, int(box[0])),
                    max(0, int(box[1])),
                    min(image_width - 1, int(box[2])),
                    min(image_height - 1, int(box[3])),
                ),
                fill=255,
            )

        def rectangle(box: tuple[float, float, float, float]) -> None:
            draw.rectangle(
                (
                    max(0, int(box[0])),
                    max(0, int(box[1])),
                    min(image_width - 1, int(box[2])),
                    min(image_height - 1, int(box[3])),
                ),
                fill=255,
            )

        if region_type is RegionType.HAT:
            # 帽冠覆盖头顶与发际线，帽檐保持在眉毛上方，避免遮住眼睛。
            ellipse(
                (
                    head_x1 - head_w * 0.08,
                    head_y1 - head_h * 0.08,
                    head_x2 + head_w * 0.08,
                    head_y1 + head_h * 0.52,
                )
            )
            ellipse(
                (
                    head_x1 - head_w * 0.16,
                    head_y1 + head_h * 0.38,
                    head_x2 + head_w * 0.16,
                    head_y1 + head_h * 0.56,
                )
            )
        elif region_type is RegionType.GLASSES:
            lens_w = head_w * 0.34
            lens_h = max(3, head_h * 0.22)
            cy = head_y1 + head_h * 0.42
            ellipse(
                (
                    head_x1 + head_w * 0.12 - lens_w / 2,
                    cy - lens_h / 2,
                    head_x1 + head_w * 0.12 + lens_w / 2,
                    cy + lens_h / 2,
                )
            )
            ellipse(
                (
                    head_x2 - head_w * 0.12 - lens_w / 2,
                    cy - lens_h / 2,
                    head_x2 - head_w * 0.12 + lens_w / 2,
                    cy + lens_h / 2,
                )
            )
            bridge_y = cy - max(1, lens_h * 0.12)
            rectangle(
                (
                    head_x1 + head_w * 0.12 + lens_w * 0.35,
                    bridge_y,
                    head_x2 - head_w * 0.12 - lens_w * 0.35,
                    bridge_y + max(2, lens_h * 0.24),
                )
            )
        elif region_type is RegionType.EARRING:
            ear_h = max(4, head_h * 0.18)
            for cx in (head_x1 + head_w * 0.03, head_x2 - head_w * 0.03):
                ellipse(
                    (
                        cx - head_w * 0.055,
                        head_y1 + head_h * 0.42,
                        cx + head_w * 0.055,
                        head_y1 + head_h * 0.42 + ear_h,
                    )
                )
        elif region_type is RegionType.HAIR:
            ellipse(
                (
                    head_x1 - head_w * 0.05,
                    head_y1,
                    head_x2 + head_w * 0.05,
                    head_y1 + head_h * 0.9,
                )
            )
            # 保留面部中心区域，避免把发型 Mask 覆盖到五官。
            draw.ellipse(
                (
                    head_x1 + head_w * 0.25,
                    head_y1 + head_h * 0.35,
                    head_x2 - head_w * 0.25,
                    head_y2,
                ),
                fill=0,
            )
        elif region_type is RegionType.TOP:
            rectangle(
                (
                    x1 + width * 0.12,
                    y1 + height * 0.22,
                    x2 - width * 0.12,
                    y1 + height * 0.58,
                )
            )
            ellipse(
                (
                    x1 + width * 0.05,
                    y1 + height * 0.2,
                    x1 + width * 0.32,
                    y1 + height * 0.34,
                )
            )
            ellipse(
                (
                    x2 - width * 0.32,
                    y1 + height * 0.2,
                    x2 - width * 0.05,
                    y1 + height * 0.34,
                )
            )
        elif region_type is RegionType.PANTS:
            rectangle(
                (
                    x1 + width * 0.22,
                    y1 + height * 0.52,
                    x2 - width * 0.22,
                    y1 + height * 0.93,
                )
            )
        elif region_type is RegionType.SHOES:
            rectangle(
                (
                    x1 + width * 0.18,
                    y1 + height * 0.9,
                    x2 - width * 0.18,
                    min(image_height - 1, y2 + height * 0.02),
                )
            )

        result = np.asarray(canvas) > 127
        if person_mask is not None and region_type in {
            RegionType.TOP,
            RegionType.PANTS,
            RegionType.SHOES,
        }:
            result = self.clip(result, person_mask, expand=4)
        if not result.any():
            raise ValueError(f"无法生成 {region_type.value} 的放置 Mask")
        return result
