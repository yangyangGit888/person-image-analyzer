"""人物穿着属性识别 Prompt 的兼容入口。

Prompt 正文已经迁移到项目根目录的 ``prompts/`` 文件，便于版本管理和独立
调整。本模块保留原有常量与 ``build_person_analysis_prompt`` 接口，避免已有
调用方和测试被破坏。
"""

from __future__ import annotations

from app.ai.prompt_manager import get_default_prompt_manager

_PROMPT_MANAGER = get_default_prompt_manager()

PERSON_ANALYSIS_SYSTEM_PROMPT = _PROMPT_MANAGER.load(
    "person_analysis_system"
).content
PERSON_ANALYSIS_PROMPT_TEMPLATE = _PROMPT_MANAGER.load(
    "person_analysis_user"
).content


def build_person_analysis_prompt(width: int, height: int) -> str:
    """根据图片尺寸生成用户分析 Prompt。"""

    return _PROMPT_MANAGER.render(
        "person_analysis_user",
        image_width=width,
        image_height=height,
    )
