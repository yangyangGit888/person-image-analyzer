#!/usr/bin/env bash
# 本地服务统一管理脚本：启动、停止、重启、状态、日志、环境自检。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

APP_MODULE="${APP_MODULE:-app.main:app}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
UVICORN_BIN="${UVICORN_BIN:-$VENV_DIR/bin/uvicorn}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/.run}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
PID_FILE="$RUN_DIR/api.pid"
LOG_FILE="$LOG_DIR/api.log"
BASE_URL="http://$HOST:$PORT"

mkdir -p "$RUN_DIR" "$LOG_DIR"

log() {
    printf '[manage] %s\n' "$*"
}

fail() {
    printf '[manage] ERROR: %s\n' "$*" >&2
    exit 1
}

read_pid() {
    if [ -f "$PID_FILE" ]; then
        tr -d '[:space:]' < "$PID_FILE"
    fi
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
    curl -fsS --max-time 3 "$BASE_URL/health" >/dev/null 2>&1
}

ensure_runtime() {
    [ -x "$PYTHON_BIN" ] || fail "未找到虚拟环境: $VENV_DIR。请先执行 ./scripts/setup.sh"
    [ -x "$UVICORN_BIN" ] || fail "未找到 uvicorn。请先执行 ./scripts/setup.sh"
}

start_service() {
    ensure_runtime
    local pid
    pid="$(read_pid || true)"
    if pid_alive "$pid"; then
        log "服务已运行，PID=$pid"
        return 0
    fi
    if [ -f "$PID_FILE" ]; then
        log "清理失效的 PID 文件"
        rm -f "$PID_FILE"
    fi
    if port_in_use; then
        fail "端口 $PORT 已被其它进程占用，请先释放端口或使用 PORT=8001 ./scripts/manage.sh start"
    fi
    if [ ! -f "$ROOT_DIR/.env" ]; then
        log "警告：未找到 .env，将使用默认配置启动"
    fi

    log "启动 API: $BASE_URL"
    nohup "$UVICORN_BIN" "$APP_MODULE" \
        --host "$HOST" \
        --port "$PORT" \
        >> "$LOG_FILE" 2>&1 &
    pid=$!
    echo "$pid" > "$PID_FILE"

    local i
    for i in $(seq 1 40); do
        if health_ok; then
            log "启动成功，PID=$pid"
            log "Swagger: $BASE_URL/docs"
            log "日志: $LOG_FILE"
            return 0
        fi
        if ! pid_alive "$pid"; then
            break
        fi
        sleep 1
    done

    log "启动失败，最近日志："
    tail -n 80 "$LOG_FILE" || true
    if pid_alive "$pid"; then
        kill "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    fail "服务未能在限定时间内就绪"
}

stop_service() {
    local pid
    pid="$(read_pid || true)"
    if ! pid_alive "$pid"; then
        rm -f "$PID_FILE"
        if port_in_use; then
            log "未找到 PID 文件，但端口 $PORT 仍被占用，请手动检查：lsof -nP -iTCP:$PORT -sTCP:LISTEN"
        else
            log "服务未运行"
        fi
        return 0
    fi

    log "停止服务 PID=$pid"
    kill -TERM "$pid" 2>/dev/null || true
    local i
    for i in $(seq 1 20); do
        if ! pid_alive "$pid"; then
            rm -f "$PID_FILE"
            log "服务已停止"
            return 0
        fi
        sleep 1
    done
    log "进程未退出，发送 SIGKILL"
    kill -KILL "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    log "服务已强制停止"
}

status_service() {
    local pid
    pid="$(read_pid || true)"
    if pid_alive "$pid"; then
        if health_ok; then
            log "运行中: PID=$pid, 健康检查正常, $BASE_URL"
        else
            log "进程存在但健康检查失败: PID=$pid"
            log "查看日志: $LOG_FILE"
            return 1
        fi
    elif port_in_use; then
        log "端口 $PORT 被占用，但没有受管 PID 文件"
        return 1
    else
        log "未运行"
        return 1
    fi
}

show_logs() {
    [ -f "$LOG_FILE" ] || fail "日志文件不存在: $LOG_FILE"
    tail -n "${LINES:-100}" -f "$LOG_FILE"
}

doctor() {
    log "项目目录: $ROOT_DIR"
    log "端口: $PORT"
    log "虚拟环境: $VENV_DIR"
    log "应用模块: $APP_MODULE"
    log "本地配套服务: 无独立数据库/Redis；SQLite 与 SAM2 均在 API 进程内"
    log "外部服务: DeepSeek API、阿里云百炼万相 API"

    if [ -x "$PYTHON_BIN" ]; then
        "$PYTHON_BIN" - <<'PY'
from app.core.config import Settings
s = Settings()
print(f"Python 配置检查: AI_PROVIDER={s.ai_provider}, AI_MODEL={s.ai_model}")
print(f"DeepSeek API Key 已配置: {s.api_key_configured}")
print(f"DashScope API Key 已配置: {s.dashscope_configured}")
print(f"编辑 Provider: {s.image_edit_provider}, 模型: {s.image_edit_model}")
print(f"分割 Provider: {s.segmentation_provider}, 设备: {s.segmentation_device}")
print(f"SAM2 权重: {s.segmentation_checkpoint}")
print(f"存储目录: {s.storage_path}")
PY
    else
        log "警告：虚拟环境不存在"
    fi

    if [ -f "$ROOT_DIR/.env" ]; then
        log ".env: 存在"
    else
        log ".env: 缺失，请复制 .env.example 并填写密钥"
    fi

    if [ -f "$ROOT_DIR/models/sam2.1_hiera_small.pt" ]; then
        log "SAM2 权重文件: 存在"
    else
        log "SAM2 权重文件: 缺失（编辑分析前需下载）"
    fi

    if [ -x "$PYTHON_BIN" ]; then
        if "$PYTHON_BIN" -c "import torch, sam2" >/dev/null 2>&1; then
            log "PyTorch/SAM2 导入: 正常"
        else
            log "PyTorch/SAM2 导入: 失败（可执行 ./scripts/setup.sh --with-ml）"
        fi
    fi

    if health_ok; then
        log "健康检查: 正常"
    elif pid_alive "$(read_pid || true)"; then
        log "健康检查: 失败"
        return 1
    else
        log "健康检查: 服务未启动"
    fi
}

case "${1:-help}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service
        start_service
        ;;
    status)
        status_service
        ;;
    logs)
        show_logs
        ;;
    doctor)
        doctor
        ;;
    help|--help|-h)
        cat <<EOF
用法: ./scripts/manage.sh <command>

命令:
  start     后台启动 API 服务
  stop      停止 API 服务
  restart   重启 API 服务
  status    查看运行状态和健康检查
  logs      实时查看日志（Ctrl+C 退出）
  doctor    检查环境、密钥配置、SAM2 权重和依赖
  help      显示本帮助

环境变量:
  HOST=127.0.0.1 PORT=8000 ./scripts/manage.sh start
  LINES=200 ./scripts/manage.sh logs
EOF
        ;;
    *)
        fail "未知命令: $1；执行 ./scripts/manage.sh help 查看用法"
        ;;
esac
