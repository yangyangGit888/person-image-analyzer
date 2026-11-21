#!/usr/bin/env bash
# 显示 API、Worker、任务队列、FFmpeg 和模型配置状态。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="http://$HOST:$PORT"
VENV_PYTHON="${VENV_DIR:-$ROOT_DIR/.venv}/bin/python"

log() { printf '[status] %s\n' "$*"; }

log "API: $BASE_URL"
if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 3 "$BASE_URL/health" >/dev/null 2>&1; then
    curl -fsS "$BASE_URL/health" | "$VENV_PYTHON" -m json.tool 2>/dev/null || curl -fsS "$BASE_URL/health"
else
    log "API 未运行或不可达"
fi

for file in api.pid worker.pid; do
    pid_file="$ROOT_DIR/.run/$file"
    if [ -f "$pid_file" ]; then
        pid="$(tr -d '[:space:]' < "$pid_file")"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            log "${file%.pid}: running pid=$pid"
        else
            log "${file%.pid}: stale pid file"
        fi
    else
        log "${file%.pid}: not running"
    fi
done

if [ -f "$ROOT_DIR/storage/database/app.sqlite3" ] && [ -x "$VENV_PYTHON" ]; then
    "$VENV_PYTHON" - <<'PY'
from app.core.config import Settings
from app.repositories.sqlite_repository import SQLiteRepository
settings = Settings()
repo = SQLiteRepository(settings.database_path)
print("[status] queue:", repo.queue_stats())
PY
fi

if command -v ffmpeg >/dev/null 2>&1; then
    log "FFmpeg: $(ffmpeg -version 2>/dev/null | head -n 1)"
elif [ -x "$ROOT_DIR/runtime/bin/ffmpeg" ]; then
    log "FFmpeg: $("$ROOT_DIR/runtime/bin/ffmpeg" -version 2>/dev/null | head -n 1)"
else
    log "FFmpeg: not found"
fi
