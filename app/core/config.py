"""应用配置模块。

所有配置均来自环境变量或项目根目录下的 ``.env`` 文件，业务代码中不硬编码
API Key、模型名称或 API 地址。使用 pydantic-settings 可以同时获得类型校验与
安全的密钥封装（SecretStr）。
"""

from __future__ import annotations

import ipaddress
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

VisionDetail = Literal["low", "high", "original", "auto"]
ReasoningEffort = Literal["low", "high", "max"]


class Settings(BaseSettings):
    """后端服务配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- 应用信息 ----------
    app_name: str = "AI 图片人物穿着属性识别系统"
    app_version: str = "1.0.0"
    app_env: Literal["development", "production", "test"] = "development"
    app_debug: bool = False
    cors_origins: str = ""

    # ---------- AI 模型配置 ----------
    ai_api_key: SecretStr = SecretStr("")
    deepseek_api_key: SecretStr = SecretStr("")
    ai_base_url: str = "https://api.deepseek.com"
    ai_model: str = "deepseek-flash"
    ai_provider: str = "deepseek"
    ai_timeout: float = Field(default=180.0, gt=0, le=600)
    ai_max_retries: int = Field(default=2, ge=0, le=5)
    ai_retry_backoff: float = Field(default=0.8, ge=0, le=10)
    ai_vision_detail: VisionDetail = "high"
    ai_enable_thinking: bool = True
    ai_reasoning_effort: ReasoningEffort = "high"
    ai_max_output_tokens: int = Field(default=16384, ge=256, le=131072)

    # 留空时由 Provider 根据已知模型列表判断；可显式设置为 true/false。
    ai_model_supports_vision: bool | None = None

    # 可选：当本地 DNS 解析结果无法连接时，直连指定 IP。
    # TLS 证书和 Host 头仍使用 AI_BASE_URL 中的域名，不会降低校验强度。
    ai_connect_ip: str | None = None

    # ---------- 图片上传配置 ----------
    max_upload_size_mb: int = Field(default=10, ge=1, le=100)
    image_max_dimension: int = Field(default=2048, ge=256, le=8192)
    image_max_pixels: int = Field(default=40_000_000, ge=1_000_000)

    # ---------- 二期能力开关（局部裁剪分析预留） ----------
    enable_crop_refinement: bool = False

    # ---------- 画像编辑：存储与任务 ----------
    storage_root: str = "./storage"
    storage_ttl_hours: int = Field(default=24, ge=1, le=24 * 30)
    edit_max_persons: int = Field(default=10, ge=1, le=50)
    edit_max_regions_per_person: int = Field(default=10, ge=1, le=30)
    edit_max_regions_per_generation: int = Field(default=5, ge=1, le=10)
    edit_min_provider_dimension: int = Field(default=512, ge=64, le=4096)
    edit_max_provider_dimension: int = Field(default=2048, ge=512, le=4096)

    # ---------- 分割模型 ----------
    segmentation_provider: Literal["sam2", "mock", "bbox"] = "sam2"
    segmentation_device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    segmentation_model_cfg: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
    segmentation_checkpoint: str = "./models/sam2.1_hiera_small.pt"
    segmentation_min_score: float = Field(default=0.05, ge=0, le=1)

    # ---------- 图像编辑模型（万相 / DashScope） ----------
    image_edit_provider: Literal["wanx", "mock"] = "wanx"
    dashscope_api_key: SecretStr = SecretStr("")
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    dashscope_workspace_id: str = ""
    image_edit_model: str = "wanx2.1-imageedit"
    image_edit_function: str = "description_edit_with_mask"
    image_edit_timeout: float = Field(default=600.0, gt=0, le=1800)
    image_edit_max_retries: int = Field(default=2, ge=0, le=5)
    image_edit_poll_initial: float = Field(default=3.0, ge=0, le=60)
    image_edit_poll_max: float = Field(default=10.0, ge=0, le=120)
    image_edit_watermark: bool = False
    image_edit_seed: int | None = None
    edit_prompt_max_length: int = Field(default=800, ge=100, le=2000)

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

    @field_validator("dashscope_base_url")
    @classmethod
    def _validate_dashscope_url(cls, value: str) -> str:
        url = value.strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError("DASHSCOPE_BASE_URL 必须以 http:// 或 https:// 开头")
        return url

    @field_validator("dashscope_workspace_id", "storage_root")
    @classmethod
    def _strip_optional_text(cls, value: str) -> str:
        return value.strip()

    @property
    def api_key(self) -> str:
        """返回明文 API Key，仅在构造请求头时使用，禁止日志输出。"""

        primary = self.ai_api_key.get_secret_value().strip()
        if primary:
            return primary
        return self.deepseek_api_key.get_secret_value().strip()

    @property
    def api_key_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def dashscope_api_key_value(self) -> str:
        return self.dashscope_api_key.get_secret_value().strip()

    @property
    def dashscope_configured(self) -> bool:
        return bool(self.dashscope_api_key_value)

    @property
    def storage_path(self) -> Path:
        return Path(self.storage_root).expanduser().resolve()

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例。"""

    return Settings()
