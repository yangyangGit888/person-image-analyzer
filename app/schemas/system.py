"""系统、健康检查与模型配置响应模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SchemaBase(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ComponentHealth(SchemaBase):
    status: Literal["ok", "degraded", "unavailable", "disabled", "unknown"] = (
        "unknown"
    )
    detail: str | None = None
    latency_ms: float | None = None


class HealthComponents(SchemaBase):
    api: ComponentHealth = Field(default_factory=ComponentHealth)
    sqlite: ComponentHealth = Field(default_factory=ComponentHealth)
    storage: ComponentHealth = Field(default_factory=ComponentHealth)
    ffmpeg: ComponentHealth = Field(default_factory=ComponentHealth)
    ai: ComponentHealth = Field(default_factory=ComponentHealth)
    worker: ComponentHealth = Field(default_factory=ComponentHealth)


class HealthResponse(SchemaBase):
    status: Literal["ok", "degraded"] = "ok"
    service: str
    version: str
    ai_configured: bool
    timestamp: datetime
    components: HealthComponents | None = None


class ModelInfo(SchemaBase):
    provider: str
    model: str
    base_url: str
    supports_vision: bool
    api_key_configured: bool
    timeout_seconds: float
    max_retries: int
    vision_formats: list[str] = Field(
        default_factory=lambda: ["jpeg", "png", "gif", "webp"]
    )
    max_image_bytes: int = 0
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    service_status: Literal["configured", "unconfigured", "unsupported_vision"] = (
        "unconfigured"
    )


class ModelInfoResponse(SchemaBase):
    success: Literal[True] = True
    message: str = "ok"
    data: ModelInfo
