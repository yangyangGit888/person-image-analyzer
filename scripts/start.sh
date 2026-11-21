#!/usr/bin/env bash
# 启动 FastAPI；视频 Worker 在 Phase 3 启用后由同一脚本启动。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
UVICORN_BIN="$VENV_DIR/bin/uvicorn"
RUN_DIR="$ROOT_DIR/.run"
LOG_DIR="$ROOT_DIR/logs"
API_PID_FILE="$RUN_DIR/api.pid"
WORKER_PID_FILE="$RUN_DIR/worker.pid"
API_LOG="$LOG_DIR/api.log"
WORKER_LOG="$LOG_DIR/worker.log"
BASE_URL="http://$HOST:$PORT"

# 允许在 .env 中关闭 Worker，但不 source .env，避免执行任意 shell 内容。
if [ -z "${VIDEO_WORKER_ENABLED:-}" ] && [ -f "$ROOT_DIR/.env" ]; then
    VIDEO_WORKER_ENABLED="$(grep -E '^VIDEO_WORKER_ENABLED=' "$ROOT_DIR/.env" | tail -n 1 | cut -d= -f2- | tr -d '[:space:]' || true)"
fi

log() { printf '[start] %s\n' "$*"; }
fail() { printf '[start] ERROR: %s\n' "$*" >&2; exit 1; }

mkdir -p "$RUN_DIR" "$LOG_DIR"
[ -x "$PYTHON_BIN" ] || fail "未找到虚拟环境，请先执行 ./scripts/setup.sh"
[ -x "$UVICORN_BIN" ] || fail "未找到 uvicorn，请先执行 ./scripts/setup.sh"

read_pid() {
    [ -f "$1" ] && tr -d '[:space:]' < "$1"
}

pid_alive() {
    local pid="${1:-}"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

port_in_use() {
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1
        return $?
    fi
    return 1
}

health_ok() {
    command -v curl >/dev/null 2>&1 && curl -fsS --max-time 3 "$BASE_URL/health" >/dev/null 2>&1
}

existing_pid="$(read_pid "$API_PID_FILE" || true)"
if pid_alive "$existing_pid"; then
    log "API 已运行，PID=$existing_pid"
    exit 0
fi
rm -f "$API_PID_FILE"
port_in_use && fail "端口 $PORT 已被其他进程占用"

log "启动 API: $BASE_URL"
nohup "$UVICORN_BIN" app.main:app --host "$HOST" --port "$PORT" >>"$API_LOG" 2>&1 &
api_pid=$!
echo "$api_pid" >"$API_PID_FILE"

for _ in $(seq 1 40); do
    if health_ok; then
        log "API 启动成功，PID=$api_pid"
        log "日志: $API_LOG"
        log "Swagger: $BASE_URL/docs"
        break
    fi
    if ! pid_alive "$api_pid"; then
        log "API 启动失败，最近日志："
        tail -n 80 "$API_LOG" || true
        rm -f "$API_PID_FILE"
        fail "API 进程已退出"
    fi
    sleep 1
done
health_ok || {
    log "API 未在限定时间内就绪，最近日志："
    tail -n 80 "$API_LOG" || true
    kill "$api_pid" 2>/dev/null || true
    rm -f "$API_PID_FILE"
    fail "API 启动超时"
}

if [ "${VIDEO_WORKER_ENABLED:-true}" = "true" ] && [ -f "$ROOT_DIR/app/workers/video_worker.py" ]; then
    existing_worker="$(read_pid "$WORKER_PID_FILE" || true)"
    if pid_alive "$existing_worker"; then
        log "Worker 已运行，PID=$existing_worker"
    else
        rm -f "$WORKER_PID_FILE"
        log "启动视频 Worker"
        nohup "$PYTHON_BIN" -m app.workers.video_worker >>"$WORKER_LOG" 2>&1 &
        worker_pid=$!
        echo "$worker_pid" >"$WORKER_PID_FILE"
        log "Worker 启动，PID=${worker_pid}，日志: $WORKER_LOG"
    fi
else
    log "视频 Worker 未启用（Phase 3）"
fi
