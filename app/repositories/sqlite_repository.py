"""SQLite 编辑仓库。

SQLite 仅作为本地嵌入式元数据存储，图片与 Mask 仍由 StorageService 保存。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from app.repositories.base import EditRepository
from app.schemas.edit import (
    EditTaskRecord,
    GenerationRecord,
    GenerationStatus,
    GenerationStepRecord,
    PersonRecord,
    RegionRecord,
    StoredImage,
    TaskStatus,
)


class SQLiteEditRepository(EditRepository):
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._db_path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS images (
                    image_id TEXT PRIMARY KEY,
                    person_count INTEGER NOT NULL DEFAULT 0,
                    analysis_status TEXT NOT NULL DEFAULT 'pending',
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    image_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_persons_image
                    ON persons(image_id, ordinal);
                CREATE TABLE IF NOT EXISTS regions (
                    region_id TEXT PRIMARY KEY,
                    image_id TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    region_type TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_regions_person
                    ON regions(person_id);
                CREATE INDEX IF NOT EXISTS idx_regions_image
                    ON regions(image_id);
                CREATE TABLE IF NOT EXISTS edit_tasks (
                    task_id TEXT PRIMARY KEY,
                    image_id TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    region_id TEXT NOT NULL,
                    task_status TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_person
                    ON edit_tasks(person_id, task_status);
                CREATE TABLE IF NOT EXISTS generations (
                    generation_id TEXT PRIMARY KEY,
                    image_id TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    plan_hash TEXT NOT NULL,
                    idempotency_key TEXT,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_generations_person
                    ON generations(person_id, status);
                CREATE INDEX IF NOT EXISTS idx_generations_plan
                    ON generations(person_id, plan_hash, status);
                CREATE TABLE IF NOT EXISTS generation_steps (
                    generation_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (generation_id, sequence)
                );
                """
            )

    # ---------- images ----------
    def save_image(self, image: StoredImage) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO images(image_id, person_count, analysis_status, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    person_count=excluded.person_count,
                    analysis_status=excluded.analysis_status,
                    payload=excluded.payload
                """,
                (
                    image.image_id,
                    image.person_count,
                    image.analysis_status.value,
                    image.model_dump_json(),
                ),
            )

    def update_image(self, image: StoredImage) -> None:
        self.save_image(image)

    def get_image(self, image_id: str) -> StoredImage | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM images WHERE image_id = ?", (image_id,)
            ).fetchone()
        return StoredImage.model_validate_json(row["payload"]) if row else None

    # ---------- persons ----------
    def save_person(self, person: PersonRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO persons(person_id, image_id, ordinal, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET
                    image_id=excluded.image_id,
                    ordinal=excluded.ordinal,
                    payload=excluded.payload
                """,
                (
                    person.person_id,
                    person.image_id,
                    person.ordinal,
                    person.model_dump_json(),
                ),
            )

    def get_person(self, person_id: str) -> PersonRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM persons WHERE person_id = ?", (person_id,)
            ).fetchone()
        return PersonRecord.model_validate_json(row["payload"]) if row else None

    def list_persons(self, image_id: str) -> list[PersonRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM persons WHERE image_id = ? ORDER BY ordinal",
                (image_id,),
            ).fetchall()
        return [PersonRecord.model_validate_json(row["payload"]) for row in rows]

    # ---------- regions ----------
    def save_region(self, region: RegionRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO regions(region_id, image_id, person_id, region_type, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(region_id) DO UPDATE SET
                    image_id=excluded.image_id,
                    person_id=excluded.person_id,
                    region_type=excluded.region_type,
                    payload=excluded.payload
                """,
                (
                    region.region_id,
                    region.image_id,
                    region.person_id,
                    region.region_type.value,
                    region.model_dump_json(),
                ),
            )

    def get_region(self, region_id: str) -> RegionRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM regions WHERE region_id = ?", (region_id,)
            ).fetchone()
        return RegionRecord.model_validate_json(row["payload"]) if row else None

    def list_regions(self, person_id: str) -> list[RegionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM regions WHERE person_id = ? ORDER BY rowid",
                (person_id,),
            ).fetchall()
        return [RegionRecord.model_validate_json(row["payload"]) for row in rows]

    def list_regions_by_image(self, image_id: str) -> list[RegionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM regions WHERE image_id = ? ORDER BY rowid",
                (image_id,),
            ).fetchall()
        return [RegionRecord.model_validate_json(row["payload"]) for row in rows]

    # ---------- tasks ----------
    def save_task(self, task: EditTaskRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO edit_tasks(
                    task_id, image_id, person_id, region_id, task_status, payload
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    image_id=excluded.image_id,
                    person_id=excluded.person_id,
                    region_id=excluded.region_id,
                    task_status=excluded.task_status,
                    payload=excluded.payload
                """,
                (
                    task.task_id,
                    task.image_id,
                    task.person_id,
                    task.region_id,
                    task.task_status.value,
                    task.model_dump_json(),
                ),
            )

    def get_task(self, task_id: str) -> EditTaskRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM edit_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return EditTaskRecord.model_validate_json(row["payload"]) if row else None

    def list_tasks(
        self,
        *,
        image_id: str | None = None,
        person_id: str | None = None,
    ) -> list[EditTaskRecord]:
        clauses: list[str] = []
        params: list[str] = []
        if image_id:
            clauses.append("image_id = ?")
            params.append(image_id)
        if person_id:
            clauses.append("person_id = ?")
            params.append(person_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT payload FROM edit_tasks {where} ORDER BY rowid",
                params,
            ).fetchall()
        return [EditTaskRecord.model_validate_json(row["payload"]) for row in rows]

    def get_task_for_region(
        self,
        person_id: str,
        region_id: str,
    ) -> EditTaskRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload FROM edit_tasks
                WHERE person_id = ? AND region_id = ? AND task_status != ?
                ORDER BY rowid DESC LIMIT 1
                """,
                (person_id, region_id, TaskStatus.CANCELLED.value),
            ).fetchone()
        return EditTaskRecord.model_validate_json(row["payload"]) if row else None

    def delete_task(self, task_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM edit_tasks WHERE task_id = ?", (task_id,))

    # ---------- generations ----------
    def save_generation(self, generation: GenerationRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO generations(
                    generation_id, image_id, person_id, status, plan_hash,
                    idempotency_key, created_at, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(generation_id) DO UPDATE SET
                    status=excluded.status,
                    plan_hash=excluded.plan_hash,
                    idempotency_key=excluded.idempotency_key,
                    payload=excluded.payload
                """,
                (
                    generation.generation_id,
                    generation.image_id,
                    generation.person_id,
                    generation.status.value,
                    generation.plan_hash,
                    generation.idempotency_key,
                    generation.created_at.isoformat(),
                    generation.model_dump_json(),
                ),
            )

    def get_generation(self, generation_id: str) -> GenerationRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM generations WHERE generation_id = ?",
                (generation_id,),
            ).fetchone()
        return GenerationRecord.model_validate_json(row["payload"]) if row else None

    def find_active_generation(
        self,
        *,
        person_id: str,
        plan_hash: str,
    ) -> GenerationRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload FROM generations
                WHERE person_id = ? AND plan_hash = ? AND status IN (?, ?)
                ORDER BY created_at DESC LIMIT 1
                """,
                (
                    person_id,
                    plan_hash,
                    GenerationStatus.PENDING.value,
                    GenerationStatus.RUNNING.value,
                ),
            ).fetchone()
        return GenerationRecord.model_validate_json(row["payload"]) if row else None

    def find_generation_by_idempotency(
        self,
        idempotency_key: str,
    ) -> GenerationRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload FROM generations
                WHERE idempotency_key = ? ORDER BY created_at DESC LIMIT 1
                """,
                (idempotency_key,),
            ).fetchone()
        return GenerationRecord.model_validate_json(row["payload"]) if row else None

    def list_active_generations(self, person_id: str) -> list[GenerationRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM generations
                WHERE person_id = ? AND status IN (?, ?)
                ORDER BY created_at
                """,
                (
                    person_id,
                    GenerationStatus.PENDING.value,
                    GenerationStatus.RUNNING.value,
                ),
            ).fetchall()
        return [GenerationRecord.model_validate_json(row["payload"]) for row in rows]

    def save_step(self, step: GenerationStepRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO generation_steps(
                    generation_id, sequence, task_id, status, payload
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(generation_id, sequence) DO UPDATE SET
                    task_id=excluded.task_id,
                    status=excluded.status,
                    payload=excluded.payload
                """,
                (
                    step.generation_id,
                    step.sequence,
                    step.task_id,
                    step.status.value,
                    step.model_dump_json(),
                ),
            )

    def update_step(self, step: GenerationStepRecord) -> None:
        self.save_step(step)

    def list_steps(self, generation_id: str) -> list[GenerationStepRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM generation_steps
                WHERE generation_id = ? ORDER BY sequence
                """,
                (generation_id,),
            ).fetchall()
        return [GenerationStepRecord.model_validate_json(row["payload"]) for row in rows]

    def mark_running_generations_interrupted(self) -> int:
        now = datetime.now().astimezone()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM generations WHERE status IN (?, ?)",
                (GenerationStatus.PENDING.value, GenerationStatus.RUNNING.value),
            ).fetchall()
            for row in rows:
                generation = GenerationRecord.model_validate_json(row["payload"])
                generation.status = GenerationStatus.INTERRUPTED
                generation.error_code = "GENERATION_INTERRUPTED"
                generation.error_detail = "服务重启导致生成任务中断，请重新提交"
                generation.updated_at = now
                connection.execute(
                    "UPDATE generations SET status = ?, payload = ? WHERE generation_id = ?",
                    (
                        generation.status.value,
                        generation.model_dump_json(),
                        generation.generation_id,
                    ),
                )
        return len(rows)

    def list_expired_image_ids(self, now: datetime) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT payload FROM images").fetchall()
        expired: list[str] = []
        for row in rows:
            image = StoredImage.model_validate_json(row["payload"])
            if image.expires_at <= now:
                expired.append(image.image_id)
        return expired

    def list_generation_ids(self, image_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT generation_id FROM generations WHERE image_id = ?",
                (image_id,),
            ).fetchall()
        return [str(row["generation_id"]) for row in rows]

    def delete_image_data(self, image_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM generation_steps WHERE generation_id IN (
                    SELECT generation_id FROM generations WHERE image_id = ?
                )
                """,
                (image_id,),
            )
            connection.execute("DELETE FROM generations WHERE image_id = ?", (image_id,))
            connection.execute("DELETE FROM edit_tasks WHERE image_id = ?", (image_id,))
            connection.execute("DELETE FROM regions WHERE image_id = ?", (image_id,))
            connection.execute("DELETE FROM persons WHERE image_id = ?", (image_id,))
            connection.execute("DELETE FROM images WHERE image_id = ?", (image_id,))
