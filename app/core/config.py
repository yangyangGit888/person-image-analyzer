"""应用配置模块。

所有配置均来自环境变量或项目根目录下的 ``.env`` 文件，业务代码中不硬编码
API Key、模型名称或 API 地址。使用 pydantic-settings 可以同时获得类型校验与
安全的密钥封装（SecretStr）。

为了兼容已有部署，AI 相关配置同时接受 ``AI_*`` 和 ``DEEPSEEK_*`` 两套
环境变量；新部署推荐使用 ``DEEPSEEK_*``。
"""

from __future__ import annotations

import ipaddress
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

VisionDetail = Literal["low", "high", "original", "auto"]
ReasoningEffort = Literal["low", "high", "max"]
MaskProviderName = Literal["bbox", "none"]


class Settings(BaseSettings):
    """后端服务配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # ---------- 应用信息 ----------
    app_name: str = "AI 图片人物穿着属性识别系统"
    app_version: str = "1.3.0"
    app_env: Literal["development", "production", "test"] = "development"
    app_debug: bool = False
    cors_origins: str = ""

    # ---------- AI 模型配置 ----------
    ai_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("AI_API_KEY", "DEEPSEEK_API_KEY"),
    )
    ai_base_url: str = Field(
        default="https://api.deepseek.com",
        validation_alias=AliasChoices("AI_BASE_URL", "DEEPSEEK_BASE_URL"),
    )
    ai_model: str = Field(
        default="deepseek-flash",
        validation_alias=AliasChoices("AI_MODEL", "DEEPSEEK_MODEL"),
    )
    ai_provider: str = "deepseek"
    ai_timeout: float = Field(
        default=180.0,
        gt=0,
        le=600,
        validation_alias=AliasChoices("AI_TIMEOUT", "DEEPSEEK_TIMEOUT"),
    )
    ai_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        validation_alias=AliasChoices("AI_MAX_RETRIES", "DEEPSEEK_MAX_RETRIES"),
    )
    ai_retry_backoff: float = Field(
        default=0.8,
        ge=0,
        le=10,
        validation_alias=AliasChoices(
            "AI_RETRY_BACKOFF",
            "DEEPSEEK_RETRY_BACKOFF",
        ),
    )
    ai_vision_detail: VisionDetail = "high"
    ai_enable_thinking: bool = True
    ai_reasoning_effort: ReasoningEffort = "high"
    ai_max_output_tokens: int = Field(default=16384, ge=256, le=131072)

    # 留空时由 Provider 根据已知模型列表判断；可显式设置为 true/false。
    ai_model_supports_vision: bool | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AI_MODEL_SUPPORTS_VISION",
            "DEEPSEEK_MODEL_SUPPORTS_VISION",
        ),
    )

    # 可选：当本地 DNS 解析结果无法连接时，直连指定 IP。
    # TLS 证书和 Host 头仍使用 AI_BASE_URL 中的域名，不会降低校验强度。
    ai_connect_ip: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AI_CONNECT_IP", "DEEPSEEK_CONNECT_IP"),
    )

    # ---------- 图片上传配置 ----------
    max_upload_size_mb: int = Field(default=10, ge=1, le=100)
    image_max_dimension: int = Field(default=2048, ge=256, le=8192)
    image_max_pixels: int = Field(default=40_000_000, ge=1_000_000)

    # 单图分析最多调用模型次数，避免多阶段分析成本失控。
    ai_max_calls_per_analysis: int = Field(default=5, ge=1, le=20)
    ai_max_concurrent_requests: int = Field(default=2, ge=1, le=20)
    analysis_crop_padding_ratio: float = Field(default=0.15, ge=0, le=0.5)
    analysis_crop_min_side: int = Field(default=512, ge=128, le=2048)
    analysis_crop_max_side: int = Field(default=1536, ge=256, le=4096)

    # ---------- 本地存储与生命周期 ----------
    storage_root: str = "./storage"
    prompts_dir: str = "./prompts"
    persist_analysis: bool = False
    analysis_retention_hours: int = Field(default=24, ge=1, le=24 * 365)
    video_retention_hours: int = Field(default=24, ge=1, le=24 * 365)
    frame_retention_hours: int = Field(default=24, ge=1, le=24 * 365)
    result_retention_days: int = Field(default=30, ge=1, le=3650)
    mask_retention_days: int = Field(default=30, ge=1, le=3650)
    cleanup_interval_seconds: int = Field(default=3600, ge=60, le=86400)

    # ---------- 视频处理 ----------
    ffmpeg_binary: str = ""
    ffprobe_binary: str = ""
    video_max_upload_size_mb: int = Field(default=100, ge=1, le=2048)
    video_max_duration_seconds: float = Field(default=20.0, gt=0, le=300)
    video_max_width: int = Field(default=1920, ge=64, le=8192)
    video_max_height: int = Field(default=1080, ge=64, le=8192)
    video_sample_fps: float = Field(default=2.0, gt=0, le=30)
    video_max_candidate_frames: int = Field(default=60, ge=1, le=600)
    video_keyframe_count: int = Field(default=8, ge=1, le=60)
    video_max_ai_calls: int = Field(default=12, ge=1, le=120)
    video_max_concurrent_tasks: int = Field(default=1, ge=1, le=10)
    video_frame_max_side: int = Field(default=1280, ge=256, le=4096)
    video_frame_jpeg_quality: int = Field(default=85, ge=50, le=100)

    # ---------- 本地帧筛选 ----------
    frame_min_sharpness: float = Field(default=50.0, ge=0)
    frame_min_brightness: float = Field(default=25.0, ge=0, le=255)
    frame_max_brightness: float = Field(default=235.0, ge=0, le=255)
    frame_dedup_hamming: int = Field(default=6, ge=0, le=32)
    frame_probe_count: int = Field(default=4, ge=1, le=20)

    # ---------- Worker ----------
    worker_lease_seconds: int = Field(default=60, ge=5, le=3600)
    worker_task_timeout_seconds: int = Field(default=600, ge=30, le=7200)
    worker_poll_interval_seconds: float = Field(default=1.0, ge=0.1, le=60)
    worker_max_attempts: int = Field(default=2, ge=1, le=10)

    # ---------- Mask ----------
    mask_provider: MaskProviderName = "bbox"

    # ---------- 二期能力开关（局部裁剪分析预留） ----------
    enable_crop_refinement: bool = False

    @field_validator("ai_provider")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        provider = value.strip().lower()
        if not provider:
            raise ValueError("AI_PROVIDER 不能为空")
        return provider

    @field_validator("ai_base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        url = value.strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError("AI_BASE_URL 必须以 http:// 或 https:// 开头")
        return url

    @field_validator("ai_model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        model = value.strip()
        if not model:
            raise ValueError("AI_MODEL 不能为空")
        return model

    @field_validator("ai_connect_ip", mode="before")
    @classmethod
    def _validate_connect_ip(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            ipaddress.ip_address(text)
        except ValueError as exc:
            raise ValueError("AI_CONNECT_IP 必须是合法的 IPv4 或 IPv6 地址") from exc
        return text

    @field_validator("storage_root", "prompts_dir", "ffmpeg_binary", "ffprobe_binary")
    @classmethod
    def _strip_path_text(cls, value: str) -> str:
        return value.strip()

    @property
    def api_key(self) -> str:
        """返回明文 API Key，仅在构造请求头时使用，禁止日志输出。"""

        return self.ai_api_key.get_secret_value().strip()

    @property
    def api_key_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def video_max_upload_size_bytes(self) -> int:
        return self.video_max_upload_size_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def storage_path(self) -> Path:
        return Path(self.storage_root).expanduser().resolve()

    @property
    def prompts_path(self) -> Path:
        return Path(self.prompts_dir).expanduser().resolve()

    @property
    def runtime_bin_path(self) -> Path:
        return self.project_root / "runtime" / "bin"

    @property
    def uploads_path(self) -> Path:
        return self.storage_path / "uploads"

    @property
    def frames_path(self) -> Path:
        return self.storage_path / "frames"

    @property
    def results_path(self) -> Path:
        return self.storage_path / "results"

    @property
    def masks_path(self) -> Path:
        return self.storage_path / "masks"

    @property
    def database_path(self) -> Path:
        return self.storage_path / "database" / "app.sqlite3"

    def _resolve_binary(self, configured: str, name: str) -> str | None:
        if configured:
            candidate = Path(configured).expanduser()
            if candidate.is_file():
                return str(candidate)
            return None
        local = self.runtime_bin_path / name
        if local.is_file():
            return str(local)
        return shutil.which(name)

    @property
    def ffmpeg_path(self) -> str | None:
        return self._resolve_binary(self.ffmpeg_binary, "ffmpeg")

    @property
    def ffprobe_path(self) -> str | None:
        return self._resolve_binary(self.ffprobe_binary, "ffprobe")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例。"""

    return Settings()
