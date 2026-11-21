"""SQLite 数据访问实现。

SQLite 只保存本地元数据和 JSON 结果；图片、视频帧、Mask 文件仍由
``StorageService`` 管理。所有写操作使用短事务，视频任务领取使用
``BEGIN IMMEDIATE`` 避免同一任务被多个 Worker 重复消费。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from app.database.schema import initialize_database
from app.schemas.analysis_v2 import AnalysisResultV2, ClothingItem
from app.schemas.mask import MaskRecord

TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "cancelled"})
RUNNING_TASK_STATUSES = (
    "preprocessing",
    "selecting_frames",
    "analyzing",
    "merging",
)
_TASK_TERMINAL = tuple(sorted(TERMINAL_TASK_STATUSES))


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    return str(value)


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


class SQLiteRepository:
    """分析结果、物品、视频任务和 Mask 元数据仓库。"""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path).expanduser().resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            initialize_database(connection)

    @property
    def db_path(self) -> Path:
        return self._db_path

    @contextmanager
    def _connect(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            if immediate:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def healthcheck(self) -> tuple[bool, str]:
        try:
            with self._connect() as connection:
                connection.execute("SELECT 1").fetchone()
            return True, "SQLite 可用"
        except sqlite3.Error:
            return False, "SQLite 不可用"

    # ---------- 分析结果与物品 ----------
    def save_analysis(
        self,
        result: AnalysisResultV2,
        *,
        source_path: str | None = None,
        canonical_path: str | None = None,
        expires_at: datetime | str | None = None,
    ) -> None:
        payload = result.model_dump(mode="json")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO analyses(
                    analysis_id, source_id, source_type, status, person_count,
                    source_path, canonical_path, created_at, expires_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(analysis_id) DO UPDATE SET
                    source_id = excluded.source_id,
                    source_type = excluded.source_type,
                    status = excluded.status,
                    person_count = excluded.person_count,
                    source_path = excluded.source_path,
                    canonical_path = excluded.canonical_path,
                    expires_at = excluded.expires_at,
                    payload = excluded.payload
                """,
                (
                    result.analysis_id,
                    result.source_id,
                    result.source_type.value,
                    result.status.value,
                    result.person_count,
                    source_path,
                    canonical_path,
                    _iso(result.created_at),
                    _iso(expires_at),
                    _json_dump(payload),
                ),
            )
            for item in result.normalized_items():
                item.analysis_id = result.analysis_id
                item.source_id = item.source_id or result.source_id
                self._save_item_conn(connection, item, source_path=source_path)

    def get_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM analyses WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()
        if row is None:
            return None
        result = _json_load(row["payload"], {})
        result["source_path"] = row["source_path"]
        result["canonical_path"] = row["canonical_path"]
        result["expires_at"] = row["expires_at"]
        return result

    def save_item(
        self,
        item: ClothingItem,
        *,
        source_path: str | None = None,
    ) -> None:
        with self._connect() as connection:
            self._save_item_conn(connection, item, source_path=source_path)

    def _save_item_conn(
        self,
        connection: sqlite3.Connection,
        item: ClothingItem,
        *,
        source_path: str | None,
    ) -> None:
        if not item.analysis_id:
            raise ValueError("物品必须关联 analysis_id")
        payload = item.model_dump(mode="json")
        bbox_json = (
            _json_dump(item.geometry.bbox)
            if item.geometry is not None and item.geometry.bbox is not None
            else None
        )
        connection.execute(
            """
            INSERT INTO items(
                item_id, analysis_id, source_id, category, type, side,
                presence, visibility, bbox_json, mask_status, mask_reference,
                source_path, created_at, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                analysis_id = excluded.analysis_id,
                source_id = excluded.source_id,
                category = excluded.category,
                type = excluded.type,
                side = excluded.side,
                presence = excluded.presence,
                visibility = excluded.visibility,
                bbox_json = excluded.bbox_json,
                mask_status = excluded.mask_status,
                mask_reference = excluded.mask_reference,
                source_path = excluded.source_path,
                payload = excluded.payload
            """,
            (
                item.item_id,
                item.analysis_id,
                item.source_id,
                item.category.value,
                item.type,
                item.side.value,
                item.presence.value,
                item.visibility.value,
                bbox_json,
                item.mask_status.value,
                item.mask_reference,
                source_path,
                _iso(item.created_at),
                _json_dump(payload),
            ),
        )

    def get_item(self, item_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM items WHERE item_id = ?",
                (item_id,),
            ).fetchone()
        if row is None:
            return None
        item = _json_load(row["payload"], {})
        item["source_path"] = row["source_path"]
        return item

    def list_items(self, analysis_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM items WHERE analysis_id = ? ORDER BY created_at",
                (analysis_id,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = _json_load(row["payload"], {})
            item["source_path"] = row["source_path"]
            result.append(item)
        return result

    # ---------- 视频任务 ----------
    def create_video_task(
        self,
        *,
        task_id: str | None = None,
        source_path: str | None = None,
        original_filename: str | None = None,
        video_info: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        usage: dict[str, int] | None = None,
        created_at: datetime | str | None = None,
        expires_at: datetime | str | None = None,
    ) -> dict[str, Any]:
        task_id = task_id or uuid.uuid4().hex
        now = _iso(created_at) or _iso(_now())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO video_tasks(
                    task_id, status, stage, progress, source_path,
                    original_filename, video_info_json, warnings_json,
                    usage_json, created_at, updated_at, expires_at
                ) VALUES (?, 'queued', 'queued', 0, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    source_path,
                    original_filename,
                    _json_dump(video_info or {}),
                    _json_dump(warnings or []),
                    _json_dump(usage or {}),
                    now,
                    now,
                    _iso(expires_at),
                ),
            )
            self._add_event_conn(connection, task_id, "queued", "queued", 0, "任务已创建")
        task = self.get_video_task(task_id)
        if task is None:
            raise RuntimeError("创建视频任务失败")
        return task

    def get_video_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM video_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return self._task_row(row) if row is not None else None

    def list_video_tasks(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(200, limit))
        offset = max(0, offset)
        query = "SELECT * FROM video_tasks"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._task_row(row) for row in rows]

    def count_video_tasks(self, *, status: str | None = None) -> int:
        query = "SELECT COUNT(*) AS total FROM video_tasks"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        return int(row["total"]) if row is not None else 0

    def update_video_task(
        self,
        task_id: str,
        **fields: Any,
    ) -> dict[str, Any] | None:
        column_map = {
            "status": "status",
            "stage": "stage",
            "progress": "progress",
            "source_path": "source_path",
            "original_filename": "original_filename",
            "error_code": "error_code",
            "error_detail": "error_detail",
            "attempts": "attempts",
            "worker_id": "worker_id",
            "cancel_requested": "cancel_requested",
            "started_at": "started_at",
            "finished_at": "finished_at",
            "heartbeat_at": "heartbeat_at",
            "expires_at": "expires_at",
            "result_analysis_id": "result_analysis_id",
        }
        json_map = {
            "video_info": "video_info_json",
            "warnings": "warnings_json",
            "usage": "usage_json",
        }
        updates: dict[str, Any] = {}
        for key, value in fields.items():
            if key in column_map:
                if isinstance(value, Enum):
                    value = value.value
                if isinstance(value, bool):
                    value = int(value)
                if isinstance(value, datetime):
                    value = _iso(value)
                updates[column_map[key]] = value
            elif key in json_map:
                updates[json_map[key]] = _json_dump(value)
        if not updates:
            return self.get_video_task(task_id)

        now = _iso(_now())
        updates["updated_at"] = now
        assignments = ", ".join(f"{column} = ?" for column in updates)
        params = list(updates.values()) + [task_id]
        with self._connect(immediate=True) as connection:
            old = connection.execute(
                "SELECT status, stage FROM video_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if old is None:
                return None
            connection.execute(
                f"UPDATE video_tasks SET {assignments} WHERE task_id = ?",
                params,
            )
            new_status = str(updates.get("status", old["status"]))
            new_stage = str(updates.get("stage", old["stage"]))
            if new_status != old["status"] or new_stage != old["stage"]:
                progress = int(updates.get("progress", 0))
                self._add_event_conn(
                    connection,
                    task_id,
                    new_status,
                    new_stage,
                    progress,
                    None,
                )
        return self.get_video_task(task_id)

    def claim_next_video_task(self, worker_id: str) -> dict[str, Any] | None:
        now = _iso(_now())
        with self._connect(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT task_id, progress FROM video_tasks
                WHERE status = 'queued' AND cancel_requested = 0
                ORDER BY created_at LIMIT 1
                """,
            ).fetchone()
            if row is None:
                return None
            task_id = str(row["task_id"])
            progress = max(5, int(row["progress"]))
            cursor = connection.execute(
                """
                UPDATE video_tasks SET
                    status = 'preprocessing',
                    stage = 'preprocessing',
                    progress = ?,
                    attempts = attempts + 1,
                    worker_id = ?,
                    started_at = COALESCE(started_at, ?),
                    heartbeat_at = ?,
                    updated_at = ?
                WHERE task_id = ? AND status = 'queued' AND cancel_requested = 0
                """,
                (progress, worker_id, now, now, now, task_id),
            )
            if cursor.rowcount != 1:
                return None
            self._add_event_conn(
                connection,
                task_id,
                "preprocessing",
                "preprocessing",
                progress,
                f"Worker {worker_id} 已领取任务",
            )
        return self.get_video_task(task_id)

    def recover_stale_video_tasks(
        self,
        *,
        lease_seconds: int,
        max_attempts: int = 2,
        now: datetime | None = None,
    ) -> dict[str, int]:
        current = now or _now()
        current_text = _iso(current)
        threshold = _iso(current - timedelta(seconds=max(1, lease_seconds)))
        recovered = 0
        failed = 0
        with self._connect(immediate=True) as connection:
            placeholders = ", ".join("?" for _ in RUNNING_TASK_STATUSES)
            rows = connection.execute(
                f"""
                SELECT task_id, attempts FROM video_tasks
                WHERE status IN ({placeholders})
                  AND (
                    (heartbeat_at IS NOT NULL AND heartbeat_at < ?)
                    OR (
                        heartbeat_at IS NULL
                        AND started_at IS NOT NULL
                        AND started_at < ?
                    )
                  )
                """,
                (*RUNNING_TASK_STATUSES, threshold, threshold),
            ).fetchall()
            for row in rows:
                task_id = str(row["task_id"])
                attempts = int(row["attempts"])
                if attempts < max_attempts:
                    connection.execute(
                        """
                        UPDATE video_tasks SET
                            status = 'queued', stage = 'queued', progress = 0,
                            worker_id = NULL, heartbeat_at = NULL,
                            error_code = 'TASK_RECOVERED',
                            error_detail = '服务重启后重新排队',
                            updated_at = ?
                        WHERE task_id = ?
                        """,
                        (current_text, task_id),
                    )
                    self._add_event_conn(
                        connection,
                        task_id,
                        "queued",
                        "queued",
                        0,
                        "服务重启后恢复任务",
                    )
                    recovered += 1
                else:
                    connection.execute(
                        """
                        UPDATE video_tasks SET
                            status = 'failed', stage = 'failed',
                            error_code = 'TASK_RECOVERED_FAILED',
                            error_detail = '超过最大重试次数，无法安全恢复',
                            finished_at = ?, updated_at = ?
                        WHERE task_id = ?
                        """,
                        (current_text, current_text, task_id),
                    )
                    self._add_event_conn(
                        connection,
                        task_id,
                        "failed",
                        "failed",
                        0,
                        "超过最大重试次数，任务失败",
                    )
                    failed += 1
        return {"recovered": recovered, "failed": failed}

    def request_cancel(self, task_id: str) -> dict[str, Any] | None:
        now = _iso(_now())
        with self._connect(immediate=True) as connection:
            row = connection.execute(
                "SELECT status FROM video_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            status = str(row["status"])
            if status not in TERMINAL_TASK_STATUSES:
                if status == "queued":
                    connection.execute(
                        """
                        UPDATE video_tasks SET
                            status = 'cancelled', stage = 'cancelled',
                            cancel_requested = 1, finished_at = ?, updated_at = ?
                        WHERE task_id = ?
                        """,
                        (now, now, task_id),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE video_tasks SET
                            cancel_requested = 1, updated_at = ?
                        WHERE task_id = ?
                        """,
                        (now, task_id),
                    )
                self._add_event_conn(
                    connection,
                    task_id,
                    "cancelled" if status == "queued" else status,
                    "cancelled" if status == "queued" else "cancelling",
                    0,
                    "收到取消请求",
                )
        return self.get_video_task(task_id)

    def queue_stats(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS total FROM video_tasks GROUP BY status"
            ).fetchall()
        return {str(row["status"]): int(row["total"]) for row in rows}

    # ---------- Mask ----------
    def save_mask(self, mask: MaskRecord) -> None:
        payload = mask.model_dump(mode="json")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO masks(
                    mask_id, item_id, analysis_id, source_id, frame_id, path,
                    width, height, method, precision, status, created_at,
                    expires_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(mask_id) DO UPDATE SET
                    status = excluded.status,
                    path = excluded.path,
                    payload = excluded.payload
                """,
                (
                    mask.mask_id,
                    mask.item_id,
                    mask.analysis_id,
                    mask.source_id,
                    mask.frame_id,
                    mask.path,
                    mask.width,
                    mask.height,
                    mask.method,
                    mask.precision.value,
                    mask.status.value,
                    _iso(mask.created_at),
                    _iso(mask.expires_at),
                    _json_dump(payload),
                ),
            )

    def get_mask(self, mask_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM masks WHERE mask_id = ?",
                (mask_id,),
            ).fetchone()
        if row is None:
            return None
        mask = _json_load(row["payload"], {})
        mask["path"] = row["path"]
        return mask

    # ---------- 生命周期 ----------
    def delete_analysis(self, analysis_id: str) -> list[str]:
        with self._connect() as connection:
            paths = self._collect_analysis_paths_conn(connection, analysis_id)
            connection.execute(
                "DELETE FROM analyses WHERE analysis_id = ?",
                (analysis_id,),
            )
        return paths

    def delete_video_task(self, task_id: str) -> list[str]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT source_path, result_analysis_id FROM video_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return []
            paths: list[str] = []
            if row["source_path"]:
                paths.append(str(row["source_path"]))
            result_analysis_id = row["result_analysis_id"]
            if result_analysis_id:
                paths.extend(
                    self._collect_analysis_paths_conn(
                        connection,
                        str(result_analysis_id),
                    )
                )
                connection.execute(
                    "DELETE FROM analyses WHERE analysis_id = ?",
                    (result_analysis_id,),
                )
            connection.execute(
                "DELETE FROM video_tasks WHERE task_id = ?",
                (task_id,),
            )
        return paths

    def cleanup_expired(
        self,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = _iso(now or _now())
        deleted_analyses = 0
        deleted_tasks = 0
        deleted_masks = 0
        paths: list[str] = []
        with self._connect(immediate=True) as connection:
            analysis_rows = connection.execute(
                "SELECT analysis_id FROM analyses WHERE expires_at IS NOT NULL AND expires_at < ?",
                (current,),
            ).fetchall()
            for row in analysis_rows:
                analysis_id = str(row["analysis_id"])
                paths.extend(self._collect_analysis_paths_conn(connection, analysis_id))
                connection.execute(
                    "DELETE FROM analyses WHERE analysis_id = ?",
                    (analysis_id,),
                )
                deleted_analyses += 1

            mask_rows = connection.execute(
                "SELECT mask_id, path FROM masks WHERE expires_at IS NOT NULL AND expires_at < ?",
                (current,),
            ).fetchall()
            for row in mask_rows:
                paths.append(str(row["path"]))
                connection.execute(
                    "DELETE FROM masks WHERE mask_id = ?",
                    (row["mask_id"],),
                )
                deleted_masks += 1

            placeholders = ", ".join("?" for _ in _TASK_TERMINAL)
            task_rows = connection.execute(
                f"""
                SELECT task_id, source_path, result_analysis_id FROM video_tasks
                WHERE expires_at IS NOT NULL AND expires_at < ?
                  AND status IN ({placeholders})
                """,
                (current, *_TASK_TERMINAL),
            ).fetchall()
            for row in task_rows:
                if row["source_path"]:
                    paths.append(str(row["source_path"]))
                result_analysis_id = row["result_analysis_id"]
                if result_analysis_id:
                    paths.extend(
                        self._collect_analysis_paths_conn(
                            connection,
                            str(result_analysis_id),
                        )
                    )
                    connection.execute(
                        "DELETE FROM analyses WHERE analysis_id = ?",
                        (result_analysis_id,),
                    )
                connection.execute(
                    "DELETE FROM video_tasks WHERE task_id = ?",
                    (row["task_id"],),
                )
                deleted_tasks += 1

        return {
            "analyses": deleted_analyses,
            "tasks": deleted_tasks,
            "masks": deleted_masks,
            "paths": list(dict.fromkeys(paths)),
        }

    # ---------- 内部辅助 ----------
    @staticmethod
    def _task_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "task_id": row["task_id"],
            "status": row["status"],
            "stage": row["stage"],
            "progress": int(row["progress"]),
            "source_path": row["source_path"],
            "original_filename": row["original_filename"],
            "video_info": _json_load(row["video_info_json"], {}),
            "error_code": row["error_code"],
            "error_detail": row["error_detail"],
            "warnings": _json_load(row["warnings_json"], []),
            "attempts": int(row["attempts"]),
            "worker_id": row["worker_id"],
            "cancel_requested": bool(row["cancel_requested"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "heartbeat_at": row["heartbeat_at"],
            "expires_at": row["expires_at"],
            "result_analysis_id": row["result_analysis_id"],
            "usage": _json_load(row["usage_json"], {}),
        }

    @staticmethod
    def _add_event_conn(
        connection: sqlite3.Connection,
        task_id: str,
        status: str,
        stage: str,
        progress: int,
        message: str | None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO task_events(task_id, status, stage, progress, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                status,
                stage,
                max(0, min(100, int(progress))),
                message,
                _iso(_now()),
            ),
        )

    @staticmethod
    def _collect_analysis_paths_conn(
        connection: sqlite3.Connection,
        analysis_id: str,
    ) -> list[str]:
        paths: list[str] = []
        row = connection.execute(
            "SELECT source_path, canonical_path FROM analyses WHERE analysis_id = ?",
            (analysis_id,),
        ).fetchone()
        if row is not None:
            paths.extend(
                str(value)
                for value in (row["source_path"], row["canonical_path"])
                if value
            )
        item_rows = connection.execute(
            "SELECT source_path FROM items WHERE analysis_id = ? AND source_path IS NOT NULL",
            (analysis_id,),
        ).fetchall()
        paths.extend(str(item["source_path"]) for item in item_rows if item["source_path"])
        mask_rows = connection.execute(
            "SELECT path FROM masks WHERE analysis_id = ?",
            (analysis_id,),
        ).fetchall()
        paths.extend(str(mask["path"]) for mask in mask_rows if mask["path"])
        return list(dict.fromkeys(paths))

