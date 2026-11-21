"""SQLite 表结构与初始化。

只使用标准库 ``sqlite3``，不引入外部数据库或 ORM。结果 JSON 原文保存在
``payload`` 字段，常用筛选字段单独建列，兼顾可查询性和未来扩展。
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
    analysis_id TEXT PRIMARY KEY,
    source_id TEXT,
    source_type TEXT NOT NULL,
    status TEXT NOT NULL,
    person_count INTEGER NOT NULL DEFAULT 0,
    source_path TEXT,
    canonical_path TEXT,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_created
    ON analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_analyses_expires
    ON analyses(expires_at);

CREATE TABLE IF NOT EXISTS items (
    item_id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL,
    source_id TEXT,
    category TEXT NOT NULL,
    type TEXT NOT NULL,
    side TEXT NOT NULL,
    presence TEXT NOT NULL,
    visibility TEXT NOT NULL,
    bbox_json TEXT,
    mask_status TEXT NOT NULL,
    mask_reference TEXT,
    source_path TEXT,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_items_analysis
    ON items(analysis_id);
CREATE INDEX IF NOT EXISTS idx_items_source
    ON items(source_id);
CREATE INDEX IF NOT EXISTS idx_items_mask_status
    ON items(mask_status);

CREATE TABLE IF NOT EXISTS video_tasks (
    task_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    source_path TEXT,
    original_filename TEXT,
    video_info_json TEXT,
    error_code TEXT,
    error_detail TEXT,
    warnings_json TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    worker_id TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    heartbeat_at TEXT,
    expires_at TEXT,
    result_analysis_id TEXT,
    usage_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_video_tasks_status
    ON video_tasks(status, created_at);
CREATE INDEX IF NOT EXISTS idx_video_tasks_expires
    ON video_tasks(expires_at);

CREATE TABLE IF NOT EXISTS task_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    message TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES video_tasks(task_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_task_events_task
    ON task_events(task_id, event_id);

CREATE TABLE IF NOT EXISTS masks (
    mask_id TEXT PRIMARY KEY,
    item_id TEXT,
    analysis_id TEXT,
    source_id TEXT,
    frame_id TEXT,
    path TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    method TEXT,
    precision TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    payload TEXT NOT NULL,
    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE,
    FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_masks_item
    ON masks(item_id);
CREATE INDEX IF NOT EXISTS idx_masks_analysis
    ON masks(analysis_id);
CREATE INDEX IF NOT EXISTS idx_masks_expires
    ON masks(expires_at);
"""


def initialize_database(connection: sqlite3.Connection) -> None:
    """创建表并记录 schema 版本。"""

    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA_SQL)
    connection.execute(
        """
        INSERT INTO schema_meta(key, value)
        VALUES('schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (str(SCHEMA_VERSION),),
    )
    connection.commit()


def get_schema_version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        return 0
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return 0
