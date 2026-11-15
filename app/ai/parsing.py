"""模型文本响应的 JSON 解析工具。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.core.exceptions import AIEmptyResponseError, AIInvalidJSONError

logger = logging.getLogger(__name__)

_FENCED_JSON_RE = re.compile(
    r"^```(?:json)?\s*(?P<body>.*?)\s*```$",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_json_content(text: str | None) -> dict[str, Any]:
    """严格解析模型返回的 JSON 对象。

    - 兼容带 ```json 代码围栏的响应（部分兼容模型会额外包裹围栏）。
    - 空内容抛出 ``AIEmptyResponseError``。
    - 非对象 JSON 或无法解析时抛出 ``AIInvalidJSONError``。
    - 错误信息与日志中不包含模型原始输出，避免潜在敏感内容泄漏。
    """

    if text is None or not text.strip():
        raise AIEmptyResponseError()

    cleaned = text.strip()
    fenced = _FENCED_JSON_RE.match(cleaned)
    if fenced:
        cleaned = fenced.group("body").strip()

    candidates = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start : end + 1])

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError) as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed
        last_error = ValueError("JSON 根节点不是对象")

    logger.warning("模型返回 JSON 解析失败: %s", type(last_error).__name__)
    raise AIInvalidJSONError() from last_error
