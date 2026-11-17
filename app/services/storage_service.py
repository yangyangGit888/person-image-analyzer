"""本地文件存储服务。

所有文件路径由服务端生成，客户端只能通过 ID 访问。写入使用临时文件 + 原子
替换，文件权限限制为当前用户。
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
    def __init__(self, settings: Settings) -> None:
        self._root = settings.storage_path
        self._root.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(self._root, 0o700)

    @property
    def root(self) -> Path:
        return self._root

    def relative_path(self, *parts: str) -> str:
        safe_parts = [part.strip("/").replace("\\", "/") for part in parts if part]
        if any(".." in part.split("/") for part in safe_parts):
            raise StorageError(detail="非法的存储路径")
        return "/".join(safe_parts)

    def resolve(self, relative_path: str) -> Path:
        candidate = (self._root / relative_path).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError as exc:
            raise StorageError(detail="非法的存储路径") from exc
        return candidate

    async def save_bytes(self, relative_path: str, data: bytes) -> str:
        await asyncio.to_thread(self._save_sync, relative_path, data)
        return relative_path

    def _save_sync(self, relative_path: str, data: bytes) -> None:
        target = self.resolve(relative_path)
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
        except OSError as exc:
            temp.unlink(missing_ok=True)
            raise StorageError(detail="无法写入本地存储") from exc

    async def read_bytes(self, relative_path: str) -> bytes:
        return await asyncio.to_thread(self._read_sync, relative_path)

    def _read_sync(self, relative_path: str) -> bytes:
        target = self.resolve(relative_path)
        try:
            return target.read_bytes()
        except OSError as exc:
            raise StorageError(detail="无法读取本地存储文件") from exc

    def exists(self, relative_path: str | None) -> bool:
        if not relative_path:
            return False
        try:
            return self.resolve(relative_path).is_file()
        except StorageError:
            return False

    def path(self, relative_path: str) -> Path:
        return self.resolve(relative_path)

    async def delete_tree(self, relative_path: str) -> None:
        await asyncio.to_thread(self._delete_tree_sync, relative_path)

    def _delete_tree_sync(self, relative_path: str) -> None:
        target = self.resolve(relative_path)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

    def new_id(self) -> str:
        return uuid.uuid4().hex
