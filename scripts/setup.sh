#!/usr/bin/env bash
# 一次性本地环境安装脚本。
# 用法:
#   ./scripts/setup.sh                 # 安装核心 + 测试依赖
#   ./scripts/setup.sh --with-ml       # 同时安装 SAM2/PyTorch 并下载权重
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
WITH_ML=false
MODEL_PATH="$ROOT_DIR/models/sam2.1_hiera_small.pt"
MODEL_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"

for arg in "$@"; do
    case "$arg" in
        --with-ml) WITH_ML=true ;;
        -h|--help)
            echo "用法: ./scripts/setup.sh [--with-ml]"
            exit 0
            ;;
        *)
            echo "未知参数: $arg" >&2
            exit 1
            ;;
    esac
done

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("需要 Python 3.11 或更高版本")
print(f"Python: {sys.version.split()[0]}")
PY

if [ ! -d "$VENV_DIR" ]; then
    echo "[setup] 创建虚拟环境: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

if [ "$WITH_ML" = true ]; then
    echo "[setup] 安装 PyTorch/SAM2 依赖"
    python -m pip install -r requirements-ml.txt
    if ! python -c "import sam2" >/dev/null 2>&1; then
        SAM2_BUILD_CUDA=0 python -m pip install "git+https://github.com/facebookresearch/sam2.git"
    fi
    mkdir -p models
    if [ ! -f "$MODEL_PATH" ]; then
        echo "[setup] 下载 SAM2.1 Hiera Small 权重"
        curl -L --fail -o "$MODEL_PATH" "$MODEL_URL"
    else
        echo "[setup] SAM2 权重已存在: $MODEL_PATH"
    fi
fi

if [ ! -f "$ROOT_DIR/.env" ]; then
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    echo "[setup] 已创建 .env，请填写 AI_API_KEY 和 DASHSCOPE_API_KEY"
else
    echo "[setup] .env 已存在，未覆盖"
fi

echo
echo "[setup] 完成。下一步："
echo "  1. 编辑 .env 填入密钥"
echo "  2. ./scripts/manage.sh doctor"
echo "  3. ./scripts/manage.sh start"
echo "  4. 打开 http://127.0.0.1:8000/docs"
