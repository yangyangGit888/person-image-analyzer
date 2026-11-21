#!/usr/bin/env bash
# 优雅停止 API 和 Worker；只处理本项目 PID 文件中的进程。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
RUN_DIR="$ROOT_DIR/.run"
log() { printf '[stop] %s\n' "$*"; }

stop_pid_file() {
    local label="$1"
    local pid_file="$2"
    local pid=""
    [ -f "$pid_file" ] && pid="$(tr -d '[:space:]' < "$pid_file")"
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$pid_file"
        log "${label} 未运行"
        return 0
    fi
    log "停止 ${label}，PID=$pid"
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 45); do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$pid_file"
            log "${label} 已停止"
            return 0
        fi
        sleep 1
    done
    log "${label} 未在 45 秒内退出，发送 SIGKILL"
    kill -KILL "$pid" 2>/dev/null || true
    rm -f "$pid_file"
}

stop_pid_file "视频 Worker" "$RUN_DIR/worker.pid"
stop_pid_file "API" "$RUN_DIR/api.pid"

if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"${PORT:-8000}" -sTCP:LISTEN >/dev/null 2>&1; then
    log "警告：端口 ${PORT:-8000} 仍被其他进程占用，请手动检查"
fi
