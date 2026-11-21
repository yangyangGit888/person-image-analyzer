# AI 图片人物穿着属性识别系统

基于 FastAPI + DeepSeek 视觉模型的后端服务。用户上传人物图片或单人短视频，服务返回结构化穿着属性。当前仓库已完成 Phase 3.1 工程收尾和 Phase 5 初始评估，正在进入 Phase 6 质量/成本优化。

- 图片分析接口 `POST /api/v1/analyze` 已可用，响应结构保持旧版兼容。
- 已引入 Prompt 外置与版本管理、V2 数据结构、坐标工具、SQLite 元数据仓库、受控本地存储、分组件健康检查和统一 `request_id`。
- 单人图片已接入 overview + 局部裁剪的多阶段分析，返回 V2 `items`、可观察性、左右脚状态和穿戴关系。
- 多人图片保留旧版多人结果，并明确标记 `unsupported_multi_person`，不伪造单人 V2 物品。
- 单人短视频已接入 FFmpeg 抽帧、质量筛选、去重、关键帧分析、SQLite 任务和本地 Worker。
- 视频结果保留关键帧 ID、时间戳、证据和冲突信息。
- 按需 Mask 将在 Phase 4 实现。
- 当前分支不使用 SAM2、YOLO、MediaPipe、Redis、Celery 或 Docker。

## 1. 技术栈

- Python 3.11+（当前验证环境 3.15.2）
- FastAPI + Uvicorn
- Pydantic v2 / pydantic-settings
- httpx（异步模型调用）
- Pillow + pillow-heif（图片校验、EXIF、HEIC/GIF）
- OpenCV（视频帧质量评估）
- SQLite（本地任务、结果、物品和 Mask 元数据）
- pytest
- python-dotenv
- FFmpeg（视频解码和抽帧，已安装 LGPL 构建到 `runtime/bin`）

## 2. 当前已实现能力

- 图片上传、真实格式校验、大小和像素限制。
- JPEG、PNG、GIF、WEBP、HEIC/HEIF；GIF 使用首帧。
- EXIF 方向校正、RGB JPEG 归一化、长边缩放。
- DeepSeek OpenAI 兼容视觉调用，base64 Data URL，`detail=high`。
- 旧版人物属性响应：头、上身、下身、腿部、脚部、三态字段。
- 单人图片 V2 多阶段分析：全图概览、下装裁剪、腿部丝袜、脚部袜子/鞋/赤脚。
- 长裙/长裤遮挡规则、左右侧规则、穿戴关系、赤脚冲突处理。
- 多人图片保留旧版逐人结果，V2 状态明确标记不支持。
- 单人短视频：ffprobe 元数据、2 FPS 抽帧、质量筛选、dHash 去重、关键帧选择。
- SQLite 视频任务、进度/阶段持久化、取消、重启恢复。
- 多帧融合、时间戳证据、冲突保留；多人视频明确失败。
- 统一错误码、超时、重试、限流、空响应、非法 JSON 处理。
- 请求级 `X-Request-ID`，错误响应包含 `request_id` 和可选 `details`。
- `/health` 分组件报告 API、SQLite、存储、FFmpeg、AI 配置和 Worker 状态。
- `/api/v1/model/info` 返回模型能力、视觉格式和 Prompt 版本，不返回密钥。
- Prompt 从 `prompts/*.txt` 加载并记录版本。
- SQLite 表结构与 Repository：分析、物品、视频任务、任务事件、Mask。
- 视频任务领取、取消标记、超时恢复和过期清理的基础能力。
- `scripts/setup.sh`、`start.sh`、`stop.sh`、`status.sh`。

## 3. 项目结构

```text
.
├── app/
│   ├── main.py                    # FastAPI 入口、lifespan、request_id、异常处理
│   ├── core/
│   │   ├── config.py              # 配置与 AI_*/DEEPSEEK_* 兼容
│   │   └── exceptions.py          # 统一错误码
│   ├── api/
│   │   ├── deps.py
│   │   └── routes/
│   │       ├── analyze.py         # POST /api/v1/analyze
│   │       ├── health.py          # GET /health
│   │       └── model.py           # GET /api/v1/model/info
│   ├── analyzers/                 # overview、下装、丝袜、袜子、鞋、赤脚
│   ├── processing/
│   │   ├── image_processor.py     # 裁剪与坐标上下文
│   │   ├── video_processor.py     # ffprobe/ffmpeg
│   │   ├── image_quality.py       # 清晰度、亮度、dHash、运动
│   │   └── frame_selector.py      # 质量筛选、去重、关键帧
│   ├── fusion/
│   │   └── video_result_fusion.py
│   ├── workers/
│   │   └── video_worker.py
│   ├── relations/
│   │   └── wearing_relation_engine.py
│   ├── observability/
│   │   └── observability_engine.py
│   ├── ai/
│   │   ├── base.py
│   │   ├── openai_compatible.py
│   │   ├── deepseek_provider.py
│   │   ├── parsing.py
│   │   ├── factory.py
│   │   └── prompt_manager.py      # Prompt 加载、版本、渲染
│   ├── prompts/
│   │   └── person_analysis.py     # 旧入口兼容，正文来自 prompts/
│   ├── schemas/
│   │   ├── analysis.py            # 旧响应结构
│   │   ├── analysis_v2.py         # V2 物品、观察性、关系、证据
│   │   ├── video.py               # 视频任务模型
│   │   ├── mask.py                # Mask 模型
│   │   └── system.py
│   ├── database/
│   │   └── schema.py              # SQLite 表结构
│   ├── repositories/
│   │   ├── base.py
│   │   └── sqlite_repository.py
│   ├── services/
│   │   ├── image_service.py
│   │   ├── analysis_service.py    # 旧版单次调用与兼容回退
│   │   ├── image_analysis_service.py
│   │   ├── video_analysis_service.py
│   │   ├── storage_service.py
│   │   └── legacy_adapter.py
│   └── utils/
│       └── coords.py              # BBox 校验与坐标映射
├── prompts/
│   ├── image_overview_system.txt
│   ├── image_overview_user.txt
│   ├── detail_system.txt
│   ├── clothing_detail_user.txt
│   ├── legs_analysis_user.txt
│   ├── feet_analysis_user.txt
│   ├── person_analysis_system.txt
│   └── person_analysis_user.txt
├── scripts/
│   ├── setup.sh
│   ├── start.sh
│   ├── stop.sh
│   ├── status.sh
│   └── test_client.py
├── tests/
├── storage/                       # 运行时生成，不提交
├── logs/                          # 运行时生成，不提交
├── runtime/bin/                   # FFmpeg/ffprobe LGPL 构建，不提交
├── ruff.toml
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

## 4. 快速开始

```bash
cd /Users/dongyang/Desktop/学习/python学习/test001

# 首次安装：创建 .venv、安装依赖、创建目录、检查 FFmpeg
./scripts/setup.sh

# 编辑 .env，填入模型凭证
# AI_API_KEY=sk-...
# 或 DEEPSEEK_API_KEY=sk-...

# 后台启动 API
./scripts/start.sh

# 查看状态
./scripts/status.sh

# 停止
./scripts/stop.sh
```

手动启动：

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Swagger：<http://127.0.0.1:8000/docs>

`setup.sh` 不会安装 torch/SAM2，也不会覆盖已有 `.env`。

## 5. 配置

### 5.1 DeepSeek

```dotenv
AI_API_KEY=sk-your-key
AI_BASE_URL=https://api.deepseek.com
AI_MODEL=deepseek-flash
AI_PROVIDER=deepseek
AI_TIMEOUT=180
AI_MAX_RETRIES=2
AI_RETRY_BACKOFF=0.8
AI_VISION_DETAIL=high
AI_ENABLE_THINKING=true
AI_REASONING_EFFORT=high
AI_MAX_OUTPUT_TOKENS=16384
```

也支持新名称：

```dotenv
DEEPSEEK_API_KEY=sk-your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_TIMEOUT=180
DEEPSEEK_MAX_RETRIES=2
```

说明：

- `deepseek-flash` 支持 JPEG、PNG、GIF、WebP。
- `deepseek-v4-pro` 不支持图片输入。
- 图片通过 base64 `data:` URL 提交，`detail: high` 等价于原图。
- 单请求体上限 48MiB，单图上限 32MiB；本项目默认上传上限 10MB。
- JSON Output 要求 Prompt 含 “json” 并提供示例；服务会处理空内容和非法 JSON。
- 未配置凭证时服务仍可启动，`/health` 的 AI 组件显示 `unavailable`。

### 5.2 存储与生命周期

```dotenv
AI_MAX_CALLS_PER_ANALYSIS=5
AI_MAX_CONCURRENT_REQUESTS=2
ANALYSIS_CROP_PADDING_RATIO=0.15
ANALYSIS_CROP_MIN_SIDE=512
ANALYSIS_CROP_MAX_SIDE=1536
STORAGE_ROOT=./storage
PROMPTS_DIR=./prompts
PERSIST_ANALYSIS=false
ANALYSIS_RETENTION_HOURS=24
VIDEO_RETENTION_HOURS=24
FRAME_RETENTION_HOURS=24
RESULT_RETENTION_DAYS=30
MASK_RETENTION_DAYS=30
```

当前图片分析仍默认不落盘；Mask 接入后会按配置持久化并清理。

### 5.3 视频与 Worker

```dotenv
FFMPEG_BINARY=
FFPROBE_BINARY=
VIDEO_MAX_DURATION_SECONDS=20
VIDEO_SAMPLE_FPS=2
VIDEO_MAX_CANDIDATE_FRAMES=60
VIDEO_MAX_AI_CALLS=12
VIDEO_MAX_CONCURRENT_TASKS=1
VIDEO_WORKER_ENABLED=true
WORKER_LEASE_SECONDS=60
WORKER_MAX_ATTEMPTS=2
```

这些参数用于视频分析，业务代码不硬编码。`start.sh` 默认同时启动 API 和 Worker。

## 6. API

### 6.1 图片分析

**POST `/api/v1/analyze`**

- `multipart/form-data`，字段 `file`。
- 支持 JPEG、PNG、GIF、WEBP、HEIC/HEIF。
- 默认最大 10MB；长边最大 2048；最大 4000 万像素。
- 响应外壳保持旧结构：

```json
{
  "success": true,
  "message": "分析成功",
  "data": {
    "image_info": {"width": 1920, "height": 1080, "person_count": 1},
    "persons": [
      {
        "person_id": 1,
        "visibility": {"legs_visible": "true", "feet_visible": "unknown"},
        "lower_body": {"wearing_pants": "true", "pants_type": "牛仔裤"},
        "legs": {"wearing_tights": "unknown"},
        "feet": {"wearing_socks": "unknown", "barefoot": "unknown"}
      }
    ]
  }
}
```

单人图片会追加 V2 字段，旧 `image_info` / `persons` 保持不变：

```json
{
  "success": true,
  "message": "分析成功",
  "data": {
    "schema_version": 2,
    "analysis_id": "...",
    "person_count": 1,
    "status": "completed",
    "items": [
      {
        "item_id": "...",
        "category": "hosiery",
        "type": "pantyhose",
        "side": "left",
        "presence": "detected",
        "visibility": "clear",
        "geometry": {
          "coordinate_space": "source_pixels",
          "bbox": [100, 200, 300, 600],
          "source_width": 1280,
          "source_height": 720,
          "precision": "estimate"
        }
      }
    ],
    "body_observability": {
      "left_foot": {"visibility": "clear", "occlusion_reason": []}
    },
    "feet": [{"side": "left", "barefoot": "not_barefoot"}],
    "wearing_relations": [
      {"type": "shoes_over_socks", "side": "left", "status": "present"}
    ],
    "warnings": []
  }
}
```

多人图片：旧 `persons` 继续返回，V2 状态为 `unsupported_multi_person`，`items` 为空，不伪造单人 V2 物品。

### 6.2 健康检查

**GET `/health`**

返回 API、SQLite、存储、FFmpeg、AI、Worker 分组件状态。AI 不可用不会让本地服务整体不可用。

```json
{
  "status": "ok",
  "service": "AI 图片人物穿着属性识别系统",
  "version": "1.3.0",
  "ai_configured": true,
  "components": {
    "api": {"status": "ok"},
    "sqlite": {"status": "ok"},
    "storage": {"status": "ok"},
    "ffmpeg": {"status": "unavailable", "detail": "未找到 ffmpeg"},
    "ai": {"status": "ok"},
    "worker": {"status": "disabled"}
  }
}
```

### 6.3 模型信息

**GET `/api/v1/model/info`**

返回 provider、model、base_url、视觉能力、上传上限、Prompt 版本和配置状态；不返回 API Key。

### 6.4 视频分析

**POST `/api/v1/analyze-video`**

- `multipart/form-data`，字段 `file`。
- 保存视频、ffprobe 校验后立即返回 `202` 和 `task_id`。
- 实际分析由本地 Worker 异步执行，HTTP 请求不会等待完整分析。

**GET `/api/v1/video-tasks/{task_id}`**

返回任务状态、阶段、进度、视频信息、错误和用量。

**GET `/api/v1/video-tasks/{task_id}/result`**

任务完成时返回融合后的 `AnalysisResultV2`；未完成、失败、取消返回明确错误码。

**POST `/api/v1/video-tasks/{task_id}/cancel`**

记录取消请求。排队任务立即取消；运行中任务在下一个检查点停止。

### 6.5 计划中的接口

- `POST /api/v1/items/{item_id}/mask`

## 7. V2 数据模型

`app/schemas/analysis_v2.py` 已定义：

- `AnalysisResultV2`：分析 ID、来源、状态、人物数、朝向、物品、观察性、脚部状态、关系、证据、警告、Prompt 版本、模型和用量。
- `ClothingItem`：`item_id`、`category`、`type`、`side`、`presence`、`visibility`、属性、风格标签、设计特征、置信等级、几何、证据、Mask 状态。
- `Geometry`：`[x_min, y_min, x_max, y_max]`、`source_pixels`、源尺寸、精度、帧 ID、时间戳。
- `BodyObservability`：下装、左右腿、左右脚。
- `WearingRelation`：丝袜覆盖脚、普通袜子叠穿、鞋覆盖袜子/丝袜。
- `FootState`：左右脚赤脚状态。
- `Evidence`：帧、时间戳、BBox、观察、置信等级、来源。

约束：

- `presence` 区分 `detected / not_detected / uncertain / not_visible`。
- 长裙遮挡时返回 `not_visible`，不能转成 `not_detected`。
- 左右侧以人物自身左右为准，无法判断时返回 `unknown`。
- 模型自报 confidence 不作为真实准确率。

## 8. Prompt 管理

Prompt 位于 `prompts/`，文件头声明版本：

```text
# prompt_version: 1
```

加载器：`app/ai/prompt_manager.py`。旧入口 `app/prompts/person_analysis.py` 保持兼容。

当前 Prompt：

- `image_overview_system` / `image_overview_user`：全图人物、朝向、下装粗识别、区域可观察性。
- `detail_system`：所有局部裁剪共用的严格规则。
- `clothing_detail_user`：下装类别、款式、风格标签、设计特征、属性和 BBox。
- `legs_analysis_user`：左右腿丝袜，连裤袜必须有连接证据。
- `feet_analysis_user`：左右脚普通袜子、鞋子、赤脚，三者独立。

## 9. 错误格式

```json
{
  "success": false,
  "message": "图片文件无效",
  "error": {
    "code": "INVALID_IMAGE",
    "detail": "图片文件已损坏或不是合法的图片",
    "details": []
  },
  "request_id": "..."
}
```

`details` 仅在存在明细时返回。所有响应带 `X-Request-ID`，请求可传入自定义 ID。

常见错误码：

- 图片：`EMPTY_FILE`、`INVALID_IMAGE`、`IMAGE_TOO_LARGE`、`UNSUPPORTED_IMAGE_FORMAT`
- AI：`AI_CONFIG_ERROR`、`AI_AUTH_ERROR`、`AI_TIMEOUT`、`AI_RATE_LIMITED`、`AI_API_ERROR`、`AI_INVALID_JSON`、`AI_EMPTY_RESPONSE`、`AI_INVALID_RESPONSE`
- 资源：`RESOURCE_NOT_FOUND`、`IMAGE_NOT_AVAILABLE`、`INVALID_BBOX`
- 视频（Phase 3）：`INVALID_VIDEO`、`UNSUPPORTED_VIDEO_FORMAT`、`VIDEO_TOO_LONG`、`VIDEO_DECODE_FAILED`、`NO_VALID_FRAMES`、`UNSUPPORTED_MULTI_PERSON`
- 任务（Phase 3）：`TASK_NOT_FOUND`、`TASK_NOT_COMPLETED`、`TASK_CANCELLED`、`TASK_TIMEOUT`
- Mask（Phase 4）：`ITEM_HAS_NO_GEOMETRY`、`MASK_PROVIDER_UNAVAILABLE`、`MASK_GENERATION_FAILED`

## 10. 测试

```bash
source .venv/bin/activate
pytest
ruff check app tests scripts
```

当前测试覆盖（108 项）：

- 旧图片上传、格式、缩放、GIF、HEIC。
- 旧结构归一化、多人、错误映射。
- DeepSeek 请求格式、重试、超时、鉴权（MockTransport，不产生费用）。
- 分组件健康检查和 `request_id`。
- Prompt 加载、版本和路径安全。
- V2 数据模型、长裙遮挡、丝袜+袜子+鞋、左右脚、赤脚。
- V2 多阶段 API：正常裤装、长裙遮挡、丝袜叠穿、单脚穿鞋、脚部不可见、多人回退、预算限制。
- SQLite 分析/物品/Mask 往返。
- 视频任务领取、完成不可重复消费、取消、超时恢复、过期清理。
- FFmpeg probe/抽帧解析、帧质量筛选、相似帧去重、关键帧时间覆盖。
- 视频融合：存在性保留、不可见不投票为不存在、冲突记录、赤脚保守融合、物品 ID 稳定。
- 视频 API 任务状态/取消/结果，Worker 完成和取消。
- 存储路径穿越和文件删除。

## 11. 识别质量评估

```bash
.venv/bin/python -m evaluation.run_evaluation --concurrency 2
```

当前 17 张人工标注图片的基线：

- Macro Precision：0.887
- Macro Recall：0.852
- Macro F1：0.837
- 下装 Category Accuracy：0.625
- 普通袜子 Type Accuracy：0.333
- 鞋子 Type Accuracy：0.182
- Side Accuracy：N/A（评估集暂无左右侧标签）
- Side Abstention Rate：0.867

报告：`evaluation/reports/latest.md`

原始图片、人工标注和预测 JSONL 默认不提交 Git；评估代码和汇总报告提交。Phase 6 第一次 Prompt 实验未通过回归，已回退；关闭 Thinking 的实验虽然降低约 72% Token，但失败率 29.4%，也未采纳。详见 `evaluation/reports/README.md`。

## 12. 阶段计划

- **Phase 1（已完成）**：配置、Prompt、V2 schema、坐标、SQLite、存储、健康检查、脚本、测试。
- **Phase 2（已完成）**：图片多阶段分析、下装/丝袜/袜子/鞋/赤脚、观察性、关系、物品 BBox，追加 V2 响应。
- **Phase 3（已完成）**：FFmpeg、Dense 抽帧、质量/去重/选帧、Worker、视频 API、多帧融合、取消与恢复。
- **Phase 3.1（已完成）**：Git、测试、代码检查、环境、配置、文档、日志和回滚点。
- **Phase 5（初始基线已完成）**：17 张人工标注图片、P/R/F1、类别/可见性/弃权率。
- **Phase 6（进行中）**：Prompt/关键帧/调用/融合优化，必须通过评估集回归。
- **Phase 4**：按需 Mask，默认粗略 BBox Mask 并明确标记 `precision=coarse`。
- **Phase 5**：部署文档、完整评估集、性能/成本测试。

## 13. 已知限制

- 单人图片已接入局部裁剪；多人图片仍使用旧版单次调用。
- V2 物品分析仅支持单人图片；多人图片明确返回兼容结果。
- Mask API 尚未启用。
- 视频仅支持单人；检测到多主要人物时任务失败并返回 `UNSUPPORTED_MULTI_PERSON`。
- 视频首版每任务默认最多 12 次模型调用，实际分析 3~6 个关键帧；不保证覆盖每一帧。
- FFmpeg 为本地最小 LGPL 构建，包含 H.264/HEVC/VP8/VP9/AV1/MPEG-4/MJPEG 解码和 image2 输出。
- 系统没有专用分割模型，Mask 只能先提供粗略矩形，不能声称像素级精确。
- 图片分析默认不落盘；Mask 需要原图，Phase 4 会启用受控持久化。

## 14. 常见问题

### 14.1 `ModuleNotFoundError: No module named 'app'`

在项目根目录启动，并激活 `.venv`。

### 14.2 `AI_AUTH_ERROR`

检查 `.env` 中 `AI_API_KEY` 或 `DEEPSEEK_API_KEY` 是否正确、是否过期、前后是否有空格。

### 14.3 `AI_TIMEOUT`

大图、多人图片和思考模式会增加耗时。可提高 `AI_TIMEOUT`，或设置 `AI_ENABLE_THINKING=false`。

### 14.4 `AI_INVALID_JSON` / `AI_EMPTY_RESPONSE`

确认 Prompt 含 “json” 和示例；输出可能被 `AI_MAX_OUTPUT_TOKENS` 截断。服务会处理空内容，但建议客户端重试。

### 14.5 FFmpeg 未找到

视频分析需要 `ffmpeg` 和 `ffprobe`。可将符合许可证要求的构建放入 `runtime/bin/`，或在 `.env` 设置 `FFMPEG_BINARY` / `FFPROBE_BINARY`。不要仅因为项目介绍是 LGPL 就认为所有编译版本都是 LGPL，分发前必须检查实际构建配置和许可证。

### 14.6 端口被占用

```bash
PORT=8001 ./scripts/start.sh
```
