"""配置别名、路径与视频默认值测试。"""

from __future__ import annotations

from pathlib import Path

from app.core.config import Settings


def test_deepseek_aliases_are_supported() -> None:
    settings = Settings(
        _env_file=None,
        **{
            "DEEPSEEK_API_KEY": "deepseek-key",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com/",
            "DEEPSEEK_MODEL": "deepseek-flash",
            "DEEPSEEK_TIMEOUT": 12,
            "DEEPSEEK_MAX_RETRIES": 1,
        },
    )

    assert settings.api_key == "deepseek-key"
    assert settings.ai_base_url == "https://api.deepseek.com"
    assert settings.ai_model == "deepseek-flash"
    assert settings.ai_timeout == 12
    assert settings.ai_max_retries == 1


def test_ai_aliases_remain_compatible() -> None:
    settings = Settings(
        _env_file=None,
        ai_api_key="legacy-key",
        ai_model="deepseek-flash",
    )

    assert settings.api_key == "legacy-key"
    assert settings.api_key_configured is True


def test_storage_and_database_paths(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        storage_root=str(tmp_path),
        prompts_dir="./prompts",
    )

    assert settings.storage_path == tmp_path.resolve()
    assert settings.uploads_path == tmp_path.resolve() / "uploads"
    assert settings.database_path == tmp_path.resolve() / "database" / "app.sqlite3"


def test_video_limits_are_configurable() -> None:
    settings = Settings(
        _env_file=None,
        video_max_duration_seconds=20,
        video_sample_fps=2,
        video_max_candidate_frames=60,
        video_max_ai_calls=12,
        video_max_concurrent_tasks=1,
    )

    assert settings.video_max_duration_seconds == 20
    assert settings.video_sample_fps == 2
    assert settings.video_max_candidate_frames == 60
    assert settings.video_max_ai_calls == 12
    assert settings.video_max_concurrent_tasks == 1


def test_ffmpeg_binary_override(tmp_path: Path) -> None:
    binary = tmp_path / "ffmpeg"
    binary.write_text("fake", encoding="utf-8")
    settings = Settings(_env_file=None, ffmpeg_binary=str(binary))

    assert settings.ffmpeg_path == str(binary)
