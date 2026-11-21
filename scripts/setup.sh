#!/usr/bin/env bash
# 一次性本地环境初始化。不会安装 torch/SAM2，也不会覆盖已有 .env。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
VENV_PYTHON="$VENV_DIR/bin/python"

log() { printf '[setup] %s\n' "$*"; }
warn() { printf '[setup] WARN: %s\n' "$*" >&2; }

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("需要 Python 3.11 或更高版本")
print(f"[setup] Python: {sys.version.split()[0]}")
PY

if [ ! -x "$VENV_PYTHON" ]; then
    log "创建虚拟环境: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

log "安装 Python 依赖"
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r requirements-dev.txt

log "创建本地目录"
mkdir -p \
    "$ROOT_DIR/storage/uploads" \
    "$ROOT_DIR/storage/frames" \
    "$ROOT_DIR/storage/results" \
    "$ROOT_DIR/storage/masks" \
    "$ROOT_DIR/storage/database" \
    "$ROOT_DIR/logs" \
    "$ROOT_DIR/.run" \
    "$ROOT_DIR/runtime/bin"
chmod 700 "$ROOT_DIR/storage" "$ROOT_DIR/logs" "$ROOT_DIR/.run" 2>/dev/null || true

if [ ! -f "$ROOT_DIR/.env" ]; then
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    log "已创建 .env，请填写模型凭证"
else
    log ".env 已存在，未覆盖"
fi

FFMPEG_CANDIDATE="${FFMPEG_BINARY:-$ROOT_DIR/runtime/bin/ffmpeg}"
if [ -x "$FFMPEG_CANDIDATE" ]; then
    log "FFmpeg: $("$FFMPEG_CANDIDATE" -version 2>/dev/null | head -n 1 || true)"
    if "$FFMPEG_CANDIDATE" -buildconf 2>/dev/null | grep -q -- '--enable-gpl'; then
        warn "当前 FFmpeg 构建启用了 GPL，分发前请更换为 LGPL 构建"
    else
        log "FFmpeg 许可证检查：未发现 --enable-gpl"
    fi
elif command -v ffmpeg >/dev/null 2>&1; then
    warn "runtime/bin 中没有 FFmpeg，将使用系统 ffmpeg；分发前请更换为符合许可证要求的构建"
    log "FFmpeg: $(ffmpeg -version 2>/dev/null | head -n 1 || true)"
else
    warn "未找到 FFmpeg/ffprobe。图片分析可继续，视频分析需要安装到 runtime/bin 或系统 PATH"
fi

if "$VENV_PYTHON" -c "import cv2" >/dev/null 2>&1; then
    log "OpenCV: $("$VENV_PYTHON" -c "import cv2; print(cv2.__version__)")"
else
    warn "OpenCV 未安装，视频帧质量评估暂不可用"
fi

log "检查应用导入"
"$VENV_PYTHON" -c "from app.main import create_app; from app.core.config import Settings; print('[setup] app import ok')"

log "完成。下一步：编辑 .env，然后执行 ./scripts/start.sh"
