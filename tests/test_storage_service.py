"""本地存储安全与生命周期基础测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.exceptions import StorageError
from app.services.storage_service import StorageService


def _storage(tmp_path: Path) -> StorageService:
    return StorageService(Settings(_env_file=None, storage_root=str(tmp_path)))


def test_storage_layout_and_roundtrip(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    relative = storage.relative_path("uploads", "demo", "image.jpg")

    storage.save_bytes(relative, b"hello")

    assert storage.exists(relative)
    assert storage.read_bytes(relative) == b"hello"
    assert storage.path(relative).is_file()
    assert (tmp_path / "uploads").is_dir()
    assert (tmp_path / "frames").is_dir()
    assert (tmp_path / "masks").is_dir()
    assert (tmp_path / "database").is_dir()


def test_storage_rejects_path_traversal(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    with pytest.raises(StorageError):
        storage.resolve("../outside.txt")
    with pytest.raises(StorageError):
        storage.save_bytes("../outside.txt", b"x")
    with pytest.raises(StorageError):
        storage.relative_path("..", "outside.txt")


def test_storage_delete_file_and_tree(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    storage.save_bytes("masks/a/mask.png", b"mask")
    storage.save_bytes("masks/a/extra.txt", b"extra")

    assert storage.delete_file("masks/a/mask.png") is True
    assert storage.exists("masks/a/mask.png") is False
    assert storage.delete_tree("masks/a") is True
    assert storage.exists("masks/a/extra.txt") is False


def test_storage_healthcheck(tmp_path: Path) -> None:
    ok, detail = _storage(tmp_path).healthcheck()
    assert ok is True
    assert detail
