"""系统与模型配置相关的响应模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class SchemaBase(BaseModel):
    model_config = ConfigDict(extra="ignore")


class HealthResponse(SchemaBase):
    status: Literal["ok"] = "ok"
    service: str
    version: str
    ai_configured: bool
    timestamp: datetime


class ModelInfo(SchemaBase):
    provider: str
    model: str
    base_url: str
    supports_vision: bool
    api_key_configured: bool
    timeout_seconds: float
    max_retries: int


class ModelInfoResponse(SchemaBase):
    success: Literal[True] = True
    message: str = "ok"
    data: ModelInfo
