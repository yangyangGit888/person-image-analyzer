"""视觉大模型统一接口。

未来接入 OpenAI、Gemini 或其它视觉模型时，只需实现 ``BaseVisionProvider``
并在 factory 中注册，业务层无需修改。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ImageInput:
    """传给模型的图片数据。

    图片仅在内存中处理，``data`` 为已归一化后的图片字节（默认 JPEG）。
    """

    data: bytes
    mime_type: str
    width: int
    height: int
    filename: str | None = None


@dataclass(slots=True)
class ModelResult:
    """模型调用的结构化结果。"""

    content: str
    parsed: dict[str, Any]
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    request_id: str | None = None


class BaseVisionProvider(ABC):
    """视觉模型 Provider 抽象基类。"""

    provider_name: str = "unknown"
    model: str = "unknown"
    base_url: str = ""
    supports_vision: bool = True

    @abstractmethod
    async def analyze_image(
        self,
        image: ImageInput,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> ModelResult:
        """分析图片并返回解析后的 JSON 结构化结果。

        实现必须保证返回值 ``parsed`` 是 dict；非 JSON 响应应抛出
        ``AIInvalidJSONError``，鉴权、超时等错误应映射为对应业务异常。
        """

    async def aclose(self) -> None:
        """释放底层 HTTP 连接，默认无操作。"""

        return None

    async def __aenter__(self) -> BaseVisionProvider:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
