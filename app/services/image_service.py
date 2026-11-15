"""图片上传校验与预处理。

职责：
- 限制上传大小，避免内存被超大文件占满。
- 通过 Pillow 解析真实图片内容，拒绝伪造后缀与损坏文件。
- 校正 EXIF 方向、统一转换为 RGB JPEG（仅内存处理，不落盘）。
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.exceptions import (
    AppError,
    EmptyFileError,
    ImageProcessingError,
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedImageFormatError,
)

register_heif_opener()

ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "HEIF", "HEIC"})
_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ProcessedImage:
    """归一化后的图片与原始信息。"""

    data: bytes
    mime_type: str
    width: int
    height: int
    source_format: str
    original_size: int
    processed_size: int


class ImageService:
    """图片读取、校验和归一化服务。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def read_upload(self, upload: UploadFile) -> bytes:
        """分块读取上传文件，并在读取过程中限制大小。"""

        limit = self._settings.max_upload_size_bytes
        declared_size = getattr(upload, "size", None)
        if isinstance(declared_size, int) and declared_size > limit:
            raise ImageTooLargeError(
                detail=f"图片大小不能超过 {self._settings.max_upload_size_mb}MB",
            )

        buffer = bytearray()
        while True:
            chunk = await upload.read(_CHUNK_SIZE)
            if not chunk:
                break
            buffer.extend(chunk)
            if len(buffer) > limit:
                raise ImageTooLargeError(
                    detail=f"图片大小不能超过 {self._settings.max_upload_size_mb}MB",
                )

        if not buffer:
            raise EmptyFileError()

        return bytes(buffer)

    async def process(self, raw: bytes) -> ProcessedImage:
        """在线程池中执行 Pillow 的同步处理，避免阻塞事件循环。"""

        if not raw:
            raise EmptyFileError()
        if len(raw) > self._settings.max_upload_size_bytes:
            raise ImageTooLargeError(
                detail=f"图片大小不能超过 {self._settings.max_upload_size_mb}MB",
            )
        return await run_in_threadpool(self._process_sync, raw)

    def _process_sync(self, raw: bytes) -> ProcessedImage:
        try:
            with Image.open(BytesIO(raw)) as probe:
                source_format = (probe.format or "").upper()
                if source_format not in ALLOWED_FORMATS:
                    raise UnsupportedImageFormatError()
                width, height = probe.size
                if width <= 0 or height <= 0:
                    raise InvalidImageError()
                if width * height > self._settings.image_max_pixels:
                    raise ImageTooLargeError(
                        detail="图片分辨率过大，请压缩后重试",
                    )
                probe.verify()

            with Image.open(BytesIO(raw)) as source:
                source.load()
                oriented = ImageOps.exif_transpose(source)
                oriented.load()
                width, height = oriented.size
                if width <= 0 or height <= 0:
                    raise InvalidImageError()

                rgb_image = self._to_rgb(oriented)
                max_dimension = self._settings.image_max_dimension
                if max(width, height) > max_dimension:
                    ratio = max_dimension / max(width, height)
                    new_size = (
                        max(1, round(width * ratio)),
                        max(1, round(height * ratio)),
                    )
                    rgb_image = rgb_image.resize(new_size, Image.Resampling.LANCZOS)

                output = BytesIO()
                rgb_image.save(
                    output,
                    format="JPEG",
                    quality=92,
                    optimize=True,
                    subsampling=0,
                )

            normalized = output.getvalue()
            if not normalized:
                raise ImageProcessingError()

            return ProcessedImage(
                data=normalized,
                mime_type="image/jpeg",
                width=width,
                height=height,
                source_format=source_format,
                original_size=len(raw),
                processed_size=len(normalized),
            )
        except AppError:
            raise
        except Image.DecompressionBombError as exc:
            raise ImageTooLargeError(
                detail="图片像素数量异常，已拒绝处理",
            ) from exc
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise InvalidImageError() from exc
        except Exception as exc:
            raise ImageProcessingError() from exc

    @staticmethod
    def _to_rgb(image: Image.Image) -> Image.Image:
        """将透明背景合成到白色背景，避免透明区域变成黑色。"""

        has_alpha = image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        )
        if has_alpha:
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.getchannel("A"))
            return background
        return image.convert("RGB")
