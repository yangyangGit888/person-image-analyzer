"""FFmpeg/ffprobe 视频探测与抽帧。"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.exceptions import (
    FrameExtractionError,
    InvalidVideoError,
    UnsupportedVideoFormatError,
    VideoDecodeError,
    VideoResolutionError,
    VideoTooLargeError,
    VideoTooLongError,
)
from app.schemas.video import VideoInfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FrameRecord:
    frame_id: str
    path: Path
    timestamp: float


@dataclass(slots=True)
class ExtractedFrames:
    frames: list[FrameRecord]
    sample_fps: float


def _parse_rate(value: str | None) -> float | None:
    if not value or "/" not in value:
        return None
    numerator, denominator = value.split("/", 1)
    try:
        num = float(numerator)
        den = float(denominator)
    except ValueError:
        return None
    if den <= 0 or num <= 0:
        return None
    return num / den


class VideoProcessor:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def ffmpeg(self) -> str:
        path = self._settings.ffmpeg_path
        if not path:
            raise VideoDecodeError(detail="未找到 ffmpeg")
        return path

    @property
    def ffprobe(self) -> str:
        path = self._settings.ffprobe_path
        if not path:
            raise VideoDecodeError(detail="未找到 ffprobe")
        return path

    async def _run(
        self,
        args: list[str],
        *,
        timeout_seconds: float,
    ) -> tuple[int, str, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise VideoDecodeError(detail="FFmpeg 可执行文件不可用") from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=max(5.0, timeout_seconds),
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise VideoDecodeError(detail="FFmpeg 处理超时") from exc
        return (
            process.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )

    async def probe(self, path: Path) -> VideoInfo:
        if not path.is_file():
            raise InvalidVideoError(detail="视频文件不存在")
        size = path.stat().st_size
        if size <= 0:
            raise InvalidVideoError()
        if size > self._settings.video_max_upload_size_bytes:
            raise VideoTooLargeError(
                detail=f"视频不能超过 {self._settings.video_max_upload_size_mb}MB",
            )

        code, stdout, stderr = await self._run(
            [
                self.ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            timeout_seconds=30,
        )
        if code != 0:
            raise InvalidVideoError(detail=(stderr or stdout or "ffprobe 失败")[:300])
        try:
            payload: dict[str, Any] = json.loads(stdout)
        except (TypeError, ValueError) as exc:
            raise InvalidVideoError(detail="ffprobe 返回非法 JSON") from exc

        streams = payload.get("streams") or []
        video_stream = next(
            (
                stream
                for stream in streams
                if isinstance(stream, dict) and stream.get("codec_type") == "video"
            ),
            None,
        )
        if video_stream is None:
            raise UnsupportedVideoFormatError(detail="文件不包含视频流")

        width = int(video_stream.get("width") or 0)
        height = int(video_stream.get("height") or 0)
        if width <= 0 or height <= 0:
            raise VideoResolutionError(detail="无法读取视频分辨率")
        if (
            width > self._settings.video_max_width
            or height > self._settings.video_max_height
        ):
            raise VideoResolutionError(
                detail=(
                    f"视频分辨率 {width}x{height} 超过限制 "
                    f"{self._settings.video_max_width}x{self._settings.video_max_height}"
                )
            )

        format_info = payload.get("format") or {}
        duration = 0.0
        for raw in (format_info.get("duration"), video_stream.get("duration")):
            try:
                duration = float(raw)
            except (TypeError, ValueError):
                continue
            if duration > 0:
                break
        if duration <= 0:
            raise InvalidVideoError(detail="无法读取视频时长")
        if duration > self._settings.video_max_duration_seconds:
            raise VideoTooLongError(
                detail=(
                    f"视频时长 {duration:.1f}s 超过限制 "
                    f"{self._settings.video_max_duration_seconds:.0f}s"
                )
            )

        codec = str(video_stream.get("codec_name") or "unknown")
        if codec not in {
            "h264",
            "hevc",
            "h265",
            "vp8",
            "vp9",
            "av1",
            "mpeg4",
            "mjpeg",
        }:
            raise UnsupportedVideoFormatError(detail=f"不支持的视频编码：{codec}")

        has_audio = any(
            isinstance(stream, dict) and stream.get("codec_type") == "audio"
            for stream in streams
        )
        return VideoInfo(
            duration_seconds=duration,
            width=width,
            height=height,
            fps=_parse_rate(video_stream.get("avg_frame_rate")),
            codec=codec,
            format_name=str(format_info.get("format_name") or "unknown"),
            size_bytes=size,
            has_audio=has_audio,
        )

    async def extract_frames(
        self,
        source: Path,
        output_dir: Path,
        *,
        info: VideoInfo,
    ) -> ExtractedFrames:
        output_dir.mkdir(parents=True, exist_ok=True)
        pattern = output_dir / "frame_%06d.jpg"
        timeout = max(60.0, info.duration_seconds * 3 + 60)
        code, stdout, stderr = await self._run(
            [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-i",
                str(source),
                "-an",
                "-sn",
                "-vf",
                f"fps={self._settings.video_sample_fps}",
                "-frames:v",
                str(self._settings.video_max_candidate_frames),
                "-q:v",
                "2",
                str(pattern),
            ],
            timeout_seconds=timeout,
        )
        if code != 0:
            raise FrameExtractionError(
                detail=(stderr or stdout or "ffmpeg 抽帧失败")[:500],
            )

        files = sorted(output_dir.glob("frame_*.jpg"))
        if not files:
            raise FrameExtractionError(detail="没有提取到任何视频帧")
        frames = [
            FrameRecord(
                frame_id=file.stem,
                path=file,
                timestamp=index / self._settings.video_sample_fps,
            )
            for index, file in enumerate(files[: self._settings.video_max_candidate_frames])
        ]
        logger.info(
            "视频抽帧完成 source=%s frames=%s fps=%s",
            source.name,
            len(frames),
            self._settings.video_sample_fps,
        )
        return ExtractedFrames(frames=frames, sample_fps=self._settings.video_sample_fps)
