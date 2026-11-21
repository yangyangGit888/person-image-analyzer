"""Prompt 文件加载与版本管理。

Prompt 统一放在项目根目录的 ``prompts/`` 下，文件头部使用注释声明版本，例如::

    # prompt_version: 1

业务代码只通过 :class:`PromptManager` 读取 Prompt，便于独立调整、对比和回滚。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from app.core.exceptions import PromptConfigError

_PROMPT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_VERSION_RE = re.compile(r"^#\s*prompt_version\s*:\s*(?P<version>\S+)\s*$")
_COMMENT_RE = re.compile(r"^#")
_PLACEHOLDER_RE = re.compile(r"__([A-Z0-9_]+)__")
_BRACE_PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


@dataclass(frozen=True, slots=True)
class Prompt:
    """已加载的 Prompt 及其版本信息。"""

    name: str
    version: str
    content: str
    path: Path

    def render(self, **values: object) -> str:
        """替换 ``__NAME__`` 或 ``{{ name }}`` 形式的占位符。"""

        rendered = self.content
        for key, value in values.items():
            text = str(value)
            rendered = rendered.replace(f"__{key.upper()}__", text)
            rendered = _BRACE_PLACEHOLDER_RE.sub(
                lambda match, key=key, text=text: text
                if match.group(1).lower() == key.lower()
                else match.group(0),
                rendered,
            )
        return rendered


class PromptManager:
    """从指定目录加载 Prompt，并在内存中缓存。"""

    def __init__(self, directory: Path | str) -> None:
        self._directory = Path(directory).expanduser().resolve()
        self._cache: dict[str, Prompt] = {}
        self._lock = Lock()

    @property
    def directory(self) -> Path:
        return self._directory

    def _validate_name(self, name: str) -> str:
        normalized = name.strip()
        if not _PROMPT_NAME_RE.fullmatch(normalized):
            raise PromptConfigError(detail=f"非法 Prompt 名称：{name}")
        return normalized

    def _parse(self, path: Path, name: str) -> Prompt:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PromptConfigError(detail=f"无法读取 Prompt 文件：{name}") from exc

        version = "unknown"
        body_lines: list[str] = []
        for line in raw.splitlines():
            match = _VERSION_RE.match(line.strip())
            if match:
                version = match.group("version")
                continue
            if _COMMENT_RE.match(line) and not body_lines:
                continue
            body_lines.append(line)

        content = "\n".join(body_lines).strip()
        if not content:
            raise PromptConfigError(detail=f"Prompt 文件为空：{name}")
        return Prompt(name=name, version=version, content=content, path=path)

    def load(self, name: str) -> Prompt:
        """加载 Prompt；同名 Prompt 会缓存。"""

        normalized = self._validate_name(name)
        with self._lock:
            cached = self._cache.get(normalized)
            if cached is not None:
                return cached

            path = self._directory / f"{normalized}.txt"
            if not path.is_file():
                raise PromptConfigError(detail=f"Prompt 文件不存在：{normalized}")
            prompt = self._parse(path, normalized)
            self._cache[normalized] = prompt
            return prompt

    def render(self, name: str, **values: object) -> str:
        return self.load(name).render(**values)

    def list_versions(self) -> dict[str, str]:
        """扫描目录，返回当前可用的 Prompt 版本。"""

        versions: dict[str, str] = {}
        if not self._directory.is_dir():
            return versions
        for path in sorted(self._directory.glob("*.txt")):
            name = path.stem
            if not _PROMPT_NAME_RE.fullmatch(name):
                continue
            try:
                versions[name] = self.load(name).version
            except PromptConfigError:
                continue
        return versions


def default_prompts_dir() -> Path:
    """项目根目录下的 ``prompts/``。"""

    return Path(__file__).resolve().parents[2] / "prompts"


def get_default_prompt_manager() -> PromptManager:
    """返回默认 PromptManager，供兼容模块使用。"""

    return PromptManager(default_prompts_dir())
