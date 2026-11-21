"""受控本地文件存储。

所有路径由服务端生成，客户端只能通过不透明 ID 访问。写入使用临时文件 +
原子替换，目录权限 0700、文件权限 0600，防止路径穿越和意外覆盖。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import uuid
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import StorageError


class StorageService:
    """本地存储目录与安全读写。"""

    def __init__(self, settings: Settings) -> None:
        self._root = settings.storage_path
        self._uploads = settings.uploads_path
        self._frames = settings.frames_path
        self._results = settings.results_path
        self._masks = settings.masks_path
        self._database = settings.database_path.parent
        self.ensure_layout()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def uploads(self) -> Path:
        return self._uploads

    @property
    def frames(self) -> Path:
        return self._frames

    @property
    def results(self) -> Path:
        return self._results

    @property
    def masks(self) -> Path:
        return self._masks

    def ensure_layout(self) -> None:
        for directory in (
            self._root,
            self._uploads,
            self._frames,
            self._results,
            self._masks,
            self._database,
        ):
            try:
                directory.mkdir(parents=True, exist_ok=True)
                with contextlib.suppress(OSError):
                    os.chmod(directory, 0o700)
            except OSError as exc:
                raise StorageError(detail="无法创建本地存储目录") from exc

    def relative_path(self, *parts: str) -> str:
        safe_parts: list[str] = []
        for part in parts:
            text = str(part).strip().strip("/").replace("\\", "/")
            if not text:
                continue
            if ".." in text.split("/"):
                raise StorageError(detail="非法的存储路径")
            safe_parts.append(text)
        if not safe_parts:
            raise StorageError(detail="存储路径不能为空")
        return "/".join(safe_parts)

    def resolve(self, relative_path: str) -> Path:
        try:
            candidate = (self._root / relative_path).resolve()
            candidate.relative_to(self._root)
        except (OSError, ValueError) as exc:
            raise StorageError(detail="非法的存储路径") from exc
        return candidate

    def path(self, relative_path: str) -> Path:
        return self.resolve(relative_path)

    def exists(self, relative_path: str | None) -> bool:
        if not relative_path:
            return False
        try:
            return self.resolve(relative_path).is_file()
        except StorageError:
            return False

    def save_bytes(self, relative_path: str, data: bytes) -> str:
        target = self.resolve(relative_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(target.parent, 0o700)
            temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            try:
                with temp.open("wb") as file_obj:
                    file_obj.write(data)
                    file_obj.flush()
                    os.fsync(file_obj.fileno())
                os.replace(temp, target)
                os.chmod(target, 0o600)
            finally:
                temp.unlink(missing_ok=True)
        except OSError as exc:
            raise StorageError(detail="无法写入本地存储") from exc
        return relative_path

    def read_bytes(self, relative_path: str) -> bytes:
        target = self.resolve(relative_path)
        try:
            return target.read_bytes()
        except OSError as exc:
            raise StorageError(detail="无法读取本地存储文件") from exc

    def delete_file(self, relative_path: str | None) -> bool:
        if not relative_path:
            return False
        try:
            target = self.resolve(relative_path)
        except StorageError:
            return False
        if not target.exists() or not target.is_file():
            return False
        try:
            target.unlink()
            return True
        except OSError:
            return False

    def delete_tree(self, relative_path: str) -> bool:
        try:
            target = self.resolve(relative_path)
        except StorageError:
            return False
        if not target.exists():
            return False
        try:
            shutil.rmtree(target, ignore_errors=False)
            return True
        except OSError:
            return False

    def healthcheck(self) -> tuple[bool, str]:
        try:
            self.ensure_layout()
        except StorageError:
            return False, "存储目录不可用"
        if not os.access(self._root, os.W_OK):
            return False, "存储目录不可写"
        return True, "存储目录可用"

    async def save_bytes_async(self, relative_path: str, data: bytes) -> str:
        return await asyncio.to_thread(self.save_bytes, relative_path, data)

    async def read_bytes_async(self, relative_path: str) -> bytes:
        return await asyncio.to_thread(self.read_bytes, relative_path)

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex
