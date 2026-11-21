"""FFmpeg 探测/抽帧与帧选择测试。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFilter

from app.core.config import Settings
from app.core.exceptions import VideoTooLongError
from app.processing.frame_selector import select_frames
from app.processing.video_processor import FrameRecord, VideoProcessor


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        storage_root=str(tmp_path),
        ffprobe_binary="/bin/echo",
        ffmpeg_binary="/bin/echo",
        **overrides,
    )


def test_probe_parses_video_info(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"x" * 1024)
    processor = VideoProcessor(_settings(tmp_path))

    async def fake_run(args, *, timeout_seconds):
        return (
            0,
            json.dumps(
                {
                    "format": {
                        "duration": "6.5",
                        "format_name": "mov,mp4",
                        "size": "1024",
                    },
                    "streams": [
                        {
                            "codec_type": "video",
                            "width": 1280,
                            "height": 720,
                            "codec_name": "h264",
                            "avg_frame_rate": "30/1",
                        },
                        {"codec_type": "audio"},
                    ],
                }
            ),
            "",
        )

    monkeypatch.setattr(processor, "_run", fake_run)
    info = asyncio.run(processor.probe(source))

    assert info.duration_seconds == 6.5
    assert info.width == 1280
    assert info.height == 720
    assert info.fps == 30
    assert info.codec == "h264"
    assert info.has_audio is True


def test_probe_rejects_too_long_video(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"x" * 1024)
    processor = VideoProcessor(
        _settings(tmp_path, video_max_duration_seconds=2)
    )

    async def fake_run(args, *, timeout_seconds):
        return (
            0,
            json.dumps(
                {
                    "format": {"duration": "10", "format_name": "mov"},
                    "streams": [
                        {
                            "codec_type": "video",
                            "width": 640,
                            "height": 480,
                            "codec_name": "h264",
                        }
                    ],
                }
            ),
            "",
        )

    monkeypatch.setattr(processor, "_run", fake_run)
    with pytest.raises(VideoTooLongError):
        asyncio.run(processor.probe(source))


def test_extract_frames_creates_records(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"x" * 1024)
    output = tmp_path / "frames"
    processor = VideoProcessor(_settings(tmp_path, video_sample_fps=2))

    async def fake_run(args, *, timeout_seconds):
        output.mkdir(parents=True, exist_ok=True)
        for index in range(1, 4):
            (output / f"frame_{index:06d}.jpg").write_bytes(b"jpeg")
        return 0, "", ""

    monkeypatch.setattr(processor, "_run", fake_run)
    extracted = asyncio.run(
        processor.extract_frames(
            source,
            output,
            info=type("Info", (), {"duration_seconds": 6.5})(),
        )
    )

    assert len(extracted.frames) == 3
    assert extracted.frames[0].timestamp == 0
    assert extracted.frames[1].timestamp == 0.5
    assert extracted.sample_fps == 2


def _frame(path: Path, index: int, *, blur: bool = False) -> FrameRecord:
    image = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(image)
    for offset in range(0, 320, 20):
        draw.line((offset, 0, 320 - offset, 240), fill="black", width=3)
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(8))
    image.save(path, format="JPEG", quality=92)
    return FrameRecord(frame_id=f"frame_{index:06d}", path=path, timestamp=index * 0.5)


def test_frame_selector_filters_and_deduplicates(tmp_path: Path) -> None:
    frames = []
    for index in range(6):
        frames.append(
            _frame(
                tmp_path / f"frame_{index:06d}.jpg",
                index,
                blur=index == 4,
            )
        )
    # 复制一张完全相同的帧，测试去重。
    duplicate = tmp_path / "frame_duplicate.jpg"
    duplicate.write_bytes(frames[0].path.read_bytes())
    frames.append(FrameRecord("frame_duplicate", duplicate, 3.0))

    settings = Settings(_env_file=None, storage_root=str(tmp_path))
    selection = select_frames(frames, settings=settings)

    assert selection.candidate_count == 7
    assert selection.selected
    ids = [item.frame_id for item in selection.selected]
    assert "frame_004" not in ids
    assert len(ids) == len(set(ids))
