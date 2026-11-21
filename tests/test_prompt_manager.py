"""Prompt 外置与版本管理测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ai.prompt_manager import PromptManager, default_prompts_dir
from app.core.exceptions import PromptConfigError
from app.prompts.person_analysis import (
    PERSON_ANALYSIS_SYSTEM_PROMPT,
    build_person_analysis_prompt,
)


def test_default_prompts_are_loaded_with_versions() -> None:
    manager = PromptManager(default_prompts_dir())
    prompt = manager.load("person_analysis_system")

    assert prompt.version == "1"
    assert "json" in prompt.content.lower()
    assert manager.list_versions()["person_analysis_system"] == "1"


def test_existing_prompt_entrypoints_remain_compatible() -> None:
    assert "json" in PERSON_ANALYSIS_SYSTEM_PROMPT.lower()
    rendered = build_person_analysis_prompt(1280, 720)
    assert "1280" in rendered
    assert "720" in rendered
    assert "__IMAGE_WIDTH__" not in rendered


def test_missing_prompt_raises(tmp_path: Path) -> None:
    manager = PromptManager(tmp_path)
    with pytest.raises(PromptConfigError):
        manager.load("missing_prompt")


def test_prompt_name_rejects_path_traversal(tmp_path: Path) -> None:
    manager = PromptManager(tmp_path)
    with pytest.raises(PromptConfigError):
        manager.load("../secret")


def test_prompt_metadata_is_not_part_of_content(tmp_path: Path) -> None:
    (tmp_path / "demo.txt").write_text(
        "# prompt_version: 7\n# name: demo\nhello __NAME__\n",
        encoding="utf-8",
    )
    prompt = PromptManager(tmp_path).load("demo")

    assert prompt.version == "7"
    assert prompt.content == "hello __NAME__"
    assert prompt.render(name="world") == "hello world"
