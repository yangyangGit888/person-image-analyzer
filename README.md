# AI 图片人物穿着属性识别与局部编辑系统

基于 FastAPI 的后端服务，包含两套能力：

1. **人物穿着属性分析**：上传图片，调用 DeepSeek 视觉模型逐人识别穿着属性，返回统一 JSON。
2. **AI 人像局部编辑**：上传人物照片，使用 DeepSeek 人物框 + 本地 SAM 2 生成 Mask，再调用阿里云百炼万相 2.1 进行局部添加或替换。

当前版本只提供后端 API，不包含前端、外部数据库或 Docker。图片默认只在服务端受控存储目录中短期保存。

## 快速启动（推荐）

本项目在本地只需要启动一个 API 进程：

- SQLite 是内嵌存储，不需要单独启动数据库。
- SAM2 在 API 进程内按需加载，不需要单独的模型服务。
- DeepSeek 与阿里云百炼万相是外部 API，不需要 Redis 或 Docker。

首次安装：

```bash
# 核心依赖 + 测试依赖
./scripts/setup.sh

# 需要本地人像分割时，同时安装 PyTorch/SAM2 并下载权重
./scripts/setup.sh --with-ml

# 编辑 .env，填入 AI_API_KEY 和 DASHSCOPE_API_KEY
./scripts/manage.sh doctor
```

日常启停：

```bash
./scripts/manage.sh start       # 后台启动 API
./scripts/manage.sh status      # 健康检查
./scripts/manage.sh logs        # 实时日志，Ctrl+C 退出
./scripts/manage.sh restart     # 重启
./scripts/manage.sh stop        # 停止
```

启动后打开 <http://127.0.0.1:8000/docs> 进行 Swagger 测试。

---

## 1. 功能概览

### 原有分析能力

- `POST /api/v1/analyze`：一次请求完成图片分析。
- 支持 JPG、JPEG、PNG、WEBP、HEIC/HEIF。
- 识别帽子、眼镜、上衣、外套、裤子、裙子、腿部、袜子、鞋子等属性。
- 三态字段使用字符串 `"true"`、`"false"`、`"unknown"`。
- 不保存分析请求的图片。

### 新增编辑能力

- 上传编辑图片并保存到受控存储目录。
- 复用 DeepSeek 分析返回的人物 bbox，为每个人物生成唯一 `person_id`。
- 使用 SAM 2.1 对人物和已有物体进行像素级分割。
- 区分 **Object Mask**（已有物体替换）和 **Placement Mask**（添加不存在物体）。
- 预设部位：帽子、眼镜、头发、耳环、上衣、裤子、鞋子。
- 预设款式和颜色白名单，前端不能提交自由 Prompt。
- 万相 2.1 异步任务提交、状态轮询、结果下载。
- 一个任务只生成一个编辑方案，多区域按顺序执行，每个区域独立 Mask 和 Prompt。
- SQLite 保存图片、人物、区域、任务和生成状态；图片/Mask/结果保存到本地文件。

当前版本不包含 Grounding DINO；人物定位来自现有 DeepSeek 分析结果的 bbox，再交给 SAM 2 分割。

---

## 2. 项目结构

```text
app/
├── main.py
├── core/
│   ├── config.py
│   └── exceptions.py
├── api/
│   ├── deps.py
│   └── routes/
│       ├── analyze.py
│       ├── health.py
│       ├── model.py
│       ├── edit_images.py
│       ├── edit_options.py
│       ├── edit_tasks.py
│       └── edit_generations.py
├── ai/
│   ├── base.py
│   ├── deepseek_provider.py
│   ├── openai_compatible.py
│   ├── factory.py
│   ├── image_edit/
│   │   ├── base.py
│   │   ├── wanx_provider.py
│   │   ├── mock_provider.py
│   │   └── factory.py
│   └── segmentation/
│       ├── base.py
│       ├── sam2_provider.py
│       ├── mock_provider.py
│       └── factory.py
├── services/
│   ├── image_service.py
│   ├── analysis_service.py
│   ├── storage_service.py
│   ├── edit_image_service.py
│   ├── edit_analysis_service.py
│   ├── region_detection_service.py
│   ├── segmentation_service.py
│   ├── mask_service.py
│   ├── prompt_builder.py
│   ├── edit_task_service.py
│   └── generation_service.py
├── repositories/
│   ├── base.py
│   └── sqlite_repository.py
├── catalogs/
│   ├── region_definitions.py
│   └── appearance_options.py
├── schemas/
│   ├── analysis.py
│   ├── edit.py
│   ├── catalog.py
│   └── system.py
└── prompts/
    ├── person_analysis.py
    └── edit_regions.py

tests/
├── test_health.py
├── test_image_upload.py
├── test_analysis.py
├── test_ai_provider.py
├── test_edit_api.py
├── test_edit_workflows.py
├── test_generation_states.py
├── test_mask_and_prompt.py
├── test_segmentation_service.py
└── test_wanx_provider.py

scripts/
├── manage.sh               # 启动/停止/重启/状态/日志/自检
├── setup.sh                # 一次性安装核心或 ML 依赖
├── test_client.py          # 原分析客户端
└── test_edit_client.py     # 编辑流程客户端

.run/                       # 运行时 PID（自动创建，已忽略）
logs/                       # 服务日志（自动创建，已忽略）
```

---

## 3. 环境准备

推荐使用统一安装脚本：

```bash
cd /Users/dongyang/Desktop/学习/python学习/test001

# 只使用原有分析功能
./scripts/setup.sh

# 使用人像局部编辑（推荐）
./scripts/setup.sh --with-ml

# 编辑 .env
cp .env.example .env
```

`setup.sh` 会：

1. 检查 Python 版本（需要 3.11+）；
2. 创建 `.venv`；
3. 安装 `requirements-dev.txt`；
4. `--with-ml` 时安装 `requirements-ml.txt`、官方 SAM2 包并下载 `sam2.1_hiera_small.pt`；
5. 如果 `.env` 不存在，从 `.env.example` 复制一份。

手动安装等价命令：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

核心依赖包含 FastAPI、Pydantic、httpx、Pillow、numpy、pillow-heif。

---

## 4. 配置 DeepSeek

在 `.env` 中配置：

```dotenv
AI_API_KEY=sk-your-deepseek-key
AI_BASE_URL=https://api.deepseek.com
AI_MODEL=deepseek-flash
AI_PROVIDER=deepseek
AI_TIMEOUT=180
AI_MAX_RETRIES=2
```

也支持 `DEEPSEEK_API_KEY` 作为 `AI_API_KEY` 的备用变量。

当前 DeepSeek 官方图片输入模型为 `deepseek-flash`；`deepseek-v4-pro` 不支持图片输入。模型名称变化时只改配置，不改业务代码。

如果本地 DNS 指向的节点不可达，可使用现有的 `AI_CONNECT_IP` 配置；TLS 证书和 Host 仍按 `AI_BASE_URL` 域名校验。

---

## 5. 配置万相 2.1

官方文档：

- 使用指南：<https://help.aliyun.com/zh/model-studio/wanx-image-edit>
- API 参考：<https://help.aliyun.com/zh/model-studio/wanx-image-edit-api-reference>

已确认的接口事实：

- 模型：`wanx2.1-imageedit`
- 地域：仅华北2（北京），必须使用北京地域 API Key
- 局部编辑 function：`description_edit_with_mask`
- 只支持异步任务
- 原图和 Mask 的宽高必须在 512～4096 之间，单文件不超过 10MB
- Mask 必须与原图分辨率完全一致
- Mask 中纯白为编辑区域，纯黑为保留区域
- 结果 URL 有效期为 24 小时，代码会立即下载保存

`.env` 配置：

```dotenv
DASHSCOPE_API_KEY=sk-your-beijing-dashscope-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1
# 可选：使用业务空间专属域名时填写
DASHSCOPE_WORKSPACE_ID=
IMAGE_EDIT_MODEL=wanx2.1-imageedit
IMAGE_EDIT_PROVIDER=wanx
IMAGE_EDIT_TIMEOUT=600
IMAGE_EDIT_MAX_RETRIES=2
IMAGE_EDIT_POLL_INITIAL=3
IMAGE_EDIT_POLL_MAX=10
IMAGE_EDIT_WATERMARK=false
```

如果使用业务空间域名，应用会自动拼接：

```text
https://{DASHSCOPE_WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/api/v1
```

请求使用 Base64 data URL，不把本地图片暴露成公网 URL。

---

## 6. 安装 SAM 2.1（本地分割，可选但编辑必需）

编辑分析需要本地 SAM 2。原有 `/api/v1/analyze` 不依赖 SAM 2，未安装时仍可正常使用。

```bash
pip install -r requirements-ml.txt
SAM2_BUILD_CUDA=0 pip install "git+https://github.com/facebookresearch/sam2.git"
mkdir -p models
curl -L -o models/sam2.1_hiera_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

`.env`：

```dotenv
SEGMENTATION_PROVIDER=sam2
SEGMENTATION_DEVICE=auto
SEGMENTATION_MODEL_CFG=configs/sam2.1/sam2.1_hiera_s.yaml
SEGMENTATION_CHECKPOINT=./models/sam2.1_hiera_small.pt
SEGMENTATION_MIN_SCORE=0.05
```

说明：

- 已在 MacBook Apple Silicon（M5 Pro、24GB、Python 3.15.2、torch 2.15、SAM2.1 Hiera Small）上实际验证 MPS 推理成功。
- SAM2 官方对 MPS 的支持仍属 preliminary。代码使用 float32，不使用 CUDA bfloat16 autocast；MPS 初始化失败会自动回退 CPU。
- 首次加载模型需要数秒，之后复用模型实例。
- 如果 MPS 出现异常，可设置：

```dotenv
SEGMENTATION_DEVICE=cpu
```

- 如果 Python 3.15 下 SAM2 依赖无法安装，可后续拆分独立 Python 3.12 ML Worker；当前分割接口已与业务层解耦，便于替换。

测试或临时不安装 SAM2 时可使用：

```dotenv
SEGMENTATION_PROVIDER=mock
```

`mock` 只用于测试和联调，不提供真实像素级质量；`bbox` 是开发降级方案，会直接使用矩形 Mask。

---

## 7. 启动与停止服务

### 7.1 推荐方式：统一管理脚本

```bash
./scripts/manage.sh start
./scripts/manage.sh status
./scripts/manage.sh logs
./scripts/manage.sh restart
./scripts/manage.sh stop
```

常用参数：

```bash
# 指定端口启动
PORT=8001 ./scripts/manage.sh start

# 查看最近 200 行日志
LINES=200 ./scripts/manage.sh logs

# 检查环境、密钥是否配置、SAM2 权重和依赖
./scripts/manage.sh doctor
```

运行时文件：

- PID：`.run/api.pid`
- 日志：`logs/api.log`
- Swagger：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

### 7.2 前台调试方式

开发调试时也可以直接运行：

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 7.3 本地配套服务说明

当前版本没有需要单独启动的数据库、Redis、消息队列或 ML Worker：

- 元数据：`storage/edit_state.sqlite3`，由 API 进程内 SQLite 访问。
- 图片、Mask、结果：`storage/` 本地文件。
- SAM2：在 API 进程内首次编辑分析时加载。
- DeepSeek / 万相：外部 HTTP API。

因此 `./scripts/manage.sh start` 会启动整套本地服务；`./scripts/manage.sh stop` 会停止整套本地服务。如果未来拆分独立 ML Worker，可在同一脚本中增加 worker 的 PID 和启动/停止逻辑。

启动时：

- 未配置 `DASHSCOPE_API_KEY` 不会阻止服务启动，但生成接口会返回配置错误。
- 未安装 SAM2 不会阻止服务启动，但编辑分析会返回 `SEGMENTATION_UNAVAILABLE`。
- 已存在的运行中生成任务在重启时会被标记为 `interrupted`。

---

## 8. 编辑 API 流程

### 8.1 上传图片

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/edit/images" \
  -F "file=@./portrait.jpg"
```

返回：

```json
{
  "success": true,
  "message": "上传成功",
  "data": {
    "image_id": "…",
    "width": 1024,
    "height": 1365,
    "original_width": 4284,
    "original_height": 5712,
    "format": "HEIC",
    "person_count": 0,
    "analysis_status": "pending"
  }
}
```

图片会被规范化为 JPEG，并保证宽高满足万相 512～2048 的限制。宽高比过大而无法同时满足限制时会返回 `EDIT_VALIDATION_ERROR`。

### 8.2 识别人物和区域

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/edit/images/{image_id}/analyze"
```

返回人物列表和每人 7 个区域：

```json
{
  "image": {"image_id": "…", "width": 1024, "height": 1365},
  "persons": [
    {
      "person_id": "…",
      "ordinal": 1,
      "bbox": {"x": 200, "y": 50, "width": 300, "height": 900},
      "mask_ready": true,
      "confidence": 0.97
    }
  ],
  "regions": [
    {
      "region_id": "…",
      "person_id": "…",
      "region_type": "hat",
      "bbox": null,
      "existence_status": "missing",
      "mask_ready": true,
      "mask_strategy": "placement",
      "detection_method": "deepseek",
      "segmentation_method": "geometric_placement",
      "mask_quality": {"area_ratio": 0.045, "valid": true},
      "attributes": {"type": "unknown", "color": "unknown"}
    }
  ]
}
```

区域类型：

- `hat`
- `glasses`
- `hair`
- `earring`
- `top`
- `pants`
- `shoes`

`existence_status`：

- `existing`：已有物体，可替换。
- `missing`：不存在，可添加。
- `unknown`：无法确认；添加时使用放置 Mask，替换需要已有物体 Mask。

### 8.3 查询人物、区域和预设选项

```bash
curl "http://127.0.0.1:8000/api/v1/edit/images/{image_id}/persons"
curl "http://127.0.0.1:8000/api/v1/edit/persons/{person_id}/regions"
curl "http://127.0.0.1:8000/api/v1/edit/appearance/options"
```

选项 API 返回部位、支持的操作、款式和颜色白名单。前端不能提交自由 Prompt。

### 8.4 创建编辑任务

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/edit/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "…",
    "person_id": "…",
    "region_type": "hat",
    "operation": "add",
    "style_id": "baseball_cap",
    "color_id": "black"
  }'
```

服务端会：

- 校验部位、操作与存在性是否匹配；
- 校验款式和颜色是否在白名单；
- 从区域记录复用 Mask；
- 生成受控 Prompt；
- 保存待执行任务。

如果部位已存在但选择 `add`，返回 `409 EDIT_CONFLICT`；应改用 `replace`。

### 8.5 提交最终生成

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/edit/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "…",
    "person_id": "…",
    "task_ids": ["…"],
    "idempotency_key": "my-plan-001"
  }'
```

返回 `202` 和 `generation_id`。多区域生成按配置顺序执行，例如：

```text
头发 → 帽子 → 眼镜 → 耳环 → 上衣 → 裤子 → 鞋子
```

### 8.6 查询状态和下载结果

```bash
curl "http://127.0.0.1:8000/api/v1/edit/generations/{generation_id}"
curl -o result.png \
  "http://127.0.0.1:8000/api/v1/edit/generations/{generation_id}/result"
```

状态：

- `pending`
- `running`
- `succeeded`
- `failed`
- `cancelled`
- `interrupted`

每个步骤包含独立状态、万相 task_id、错误码和结果是否就绪。

### 8.7 修改和删除任务

```bash
curl -X PATCH "http://127.0.0.1:8000/api/v1/edit/tasks/{task_id}" \
  -H "Content-Type: application/json" \
  -d '{"style_id": "beret", "color_id": "red"}'

curl -X DELETE "http://127.0.0.1:8000/api/v1/edit/tasks/{task_id}"
```

---

## 9. 编辑测试客户端

```bash
source .venv/bin/activate
python scripts/test_edit_client.py ./portrait.jpg

# 指定部位和选项
python scripts/test_edit_client.py ./portrait.jpg \
  --region hat --operation add --style beret --color red \
  --output result.png
```

脚本会依次上传、分析、创建任务、提交生成、轮询并保存结果。它不会绕过服务端白名单，也不会发送自定义 Prompt。

---

## 10. Mask 策略

### Object Mask（已有物体）

1. DeepSeek 区域分析给出粗略 bbox 和存在性。
2. SAM2 使用矩形框提示分割。
3. 小目标（耳环、眼镜等）会先局部裁剪放大再分割，再映射回全图。
4. 对上衣、裤子、鞋子等与人物强相关的区域，用人物 Mask 约束并排除其他人物 Mask。
5. 校验面积、尺寸、与人物区域重叠率。
6. 保存为严格二值 PNG。

DeepSeek 的 bbox 只作为提示，不作为最终 Mask。

### Placement Mask（不存在物体）

SAM2 不能凭空生成不存在的物体，因此使用人物 bbox 与头部几何估算：

- 帽子：优先使用 SAM2 人物 Mask 的头顶和头部宽度，生成帽冠 + 帽檐，帽檐约束在眉毛上方，避开眼睛。
- 眼镜：双眼高度的镜片区域 + 鼻梁连接区。
- 耳环：左右耳部下方小椭圆。
- 上衣：肩部和躯干区域，受人物 Mask 约束。
- 裤子：髋部到脚踝区域，受人物 Mask 约束。
- 鞋子：脚部区域，受人物 Mask 约束。

当前版本使用几何估算，没有引入人脸关键点模型。后续可在 `PlacementMaskBuilder` 前增加 MediaPipe Face/Pose Landmarker，提高帽子、眼镜、耳环的贴合度。

### Mask 约束

- 尺寸必须与当前编辑图片完全一致。
- 提交万相前严格二值化：0 或 255。
- 边缘羽化只允许用于本地预览，不提交给万相。
- 面积过小、过大、空 Mask、与人物区域重叠过低均判为无效。
- Mask 文件保存在 `STORAGE_ROOT/images/{image_id}/regions/`。

---

## 11. 多区域编辑

当前推荐并实现的是**顺序执行**：

1. 校验同一人物、同一图片的所有任务。
2. 按部位依赖顺序排序。
3. 每个部位单独调用万相一次，使用该部位独立的 Mask 和 Prompt。
4. 将上一步结果作为下一步输入。
5. 每一步校验输出宽高比和尺寸，保存中间结果。
6. 将万相结果按当前 Mask 与上一步输入合成，只回填 Mask 内区域；Mask 外的脸部、眼镜、手和背景保持原图，避免人物身份漂移。
7. 全部成功后保存最终结果。

没有把多个不同语义的 Mask 合并成一次调用，因为万相每次只接受一个 Mask 和一个 Prompt，合并会丢失各区域的编辑语义。

默认单次生成最多 5 个部位，避免费用和耗时失控。失败会保留已完成步骤和中间图片，生成状态为 `failed`，错误码记录在对应步骤和任务上。

---

## 12. 数据模型与本地存储

### 数据实体

- **Image**：图片 ID、原图尺寸、规范化尺寸、格式、文件大小、分析状态、过期时间。
- **Person**：人物 ID、图片 ID、序号、像素 bbox、人物 Mask、置信度。
- **Region**：区域 ID、人物 ID、部位类型、bbox、Mask、检测方法、分割方法、存在性、Mask 质量、属性。
- **EditTask**：任务 ID、图片/人物/区域、操作、选中选项、自动 Prompt、Mask、状态、万相任务 ID、结果。
- **Generation**：生成 ID、图片/人物、任务顺序、状态、当前步骤、结果路径、错误。
- **GenerationStep**：每一步的顺序、万相任务 ID、输入/输出图片、状态和错误。

一个图片可包含多个 Person；一个 Person 可包含多个 Region；一个 Image 可包含多个 EditTask 和 Generation。

### 存储

```text
storage/
├── edit_state.sqlite3
├── images/{image_id}/
│   ├── source.jpg
│   ├── persons/{person_id}.png
│   └── regions/{region_id}.png
└── generations/{generation_id}/
    ├── step_01.png
    └── result.png
```

- 路径由服务端 UUID 生成，客户端不能提交文件路径。
- 图片、Mask 和结果不挂公开静态目录。
- 默认保留 24 小时，`STORAGE_TTL_HOURS` 可调整。
- SQLite 是标准库嵌入式存储，不引入外部数据库服务。
- 服务启动时会清理已过期（`STORAGE_TTL_HOURS`）的图片、Mask 和生成结果；后续可增加定时清理任务或管理命令。

---

## 13. 安全与隐私

- API Key 只从环境变量读取，不返回给客户端。
- 日志不会输出 API Key；万相错误日志会做脱敏。
- 请求模型使用 `extra="forbid"`，自由 Prompt 字段会被拒绝。
- 编辑选项和 Prompt 片段全部来自服务端 catalog。
- 图片发送万相时使用 Base64，不使用公网图片 URL。
- 万相结果 URL 只在服务端短时间使用，立即下载并保存到本地。
- 存储根目录权限限制为当前用户。
- 当前项目没有用户鉴权；如果要部署到公网，应在反向代理或 FastAPI 层增加鉴权、限流和存储清理。
- 用户图片会上传到 DeepSeek 和阿里云百炼，请根据实际隐私要求确认合规性。

---

## 14. 运行测试

```bash
source .venv/bin/activate
pytest
ruff check app tests scripts
```

测试不调用真实 DeepSeek、万相或 SAM2 权重：

- 使用 FakeProvider 测试人物分析和区域定位。
- 使用 `httpx.MockTransport` 测试万相请求格式、鉴权、内容审核、状态和下载。
- 使用 MockSegmentationProvider 测试人物/区域 Mask。
- 使用 MockImageEditProvider 测试生成成功、多区域顺序、失败、超时、幂等。
- 现有分析测试全部保留。

当前测试覆盖：

- 健康检查、模型配置不泄漏 Key
- JPG/PNG/WEBP/HEIC 上传
- 损坏图片、空文件、超大文件、GIF 拒绝
- 多人物独立结果
- 三态值归一化
- 编辑图片上传与规范化
- 7 个部位的存在性、Mask 策略和操作冲突
- 添加帽子、替换帽子、添加眼镜、多区域生成
- 多人照片区域关联
- 万相 Mock 请求、鉴权、内容拒绝、下载
- 生成失败、超时、重复提交
- 自由 Prompt 拒绝

真实 SAM2 推理需要本地权重，可作为单独的手工冒烟测试：

```bash
SEGMENTATION_PROVIDER=sam2 \
SEGMENTATION_DEVICE=auto \
SEGMENTATION_CHECKPOINT=./models/sam2.1_hiera_small.pt \
python scripts/test_edit_client.py ./portrait.jpg
```

真实万相调用会消耗费用，需配置北京地域 `DASHSCOPE_API_KEY` 后手动执行。

本项目已在真实图片上验证：

- 添加黑色棒球帽；
- 将白色衬衫替换为黑色衬衫；
- 同一次生成中顺序执行“添加帽子 + 替换上衣”；
- DeepSeek 人物框 + SAM2 MPS 分割 + 万相 2.1 异步任务 + 结果下载 + Mask 回填全部成功。

万相在测试时段单次任务排队约数分钟，因此默认 `IMAGE_EDIT_TIMEOUT=600`；多区域顺序生成会按区域累加耗时。

---

## 15. macOS Apple Silicon 说明

- 当前开发机：Apple M5 Pro，24GB，Python 3.15.2。
- 已验证 torch 2.15 + SAM2.1 Hiera Small 在 MPS 上完成图像分割。
- 若 MPS 报错，设置 `SEGMENTATION_DEVICE=cpu`。
- SAM2 首次加载和首次推理较慢，之后会复用模型。
- Python 3.15 下 SAM2 可能出现 `torch.jit.script` 的 FutureWarning，这是依赖链的警告，不影响当前推理。
- 模型权重不提交 Git，需按第 6 节下载。
- 若 Python 3.15 下依赖无法安装，建议使用 Python 3.12 虚拟环境运行 ML 部分，主 API 与分割接口已经解耦。

---

## 16. 常见问题

### 16.1 `SEGMENTATION_UNAVAILABLE`

未安装 SAM2/PyTorch 或权重路径错误。执行：

```bash
pip install -r requirements-ml.txt
SAM2_BUILD_CUDA=0 pip install "git+https://github.com/facebookresearch/sam2.git"
ls -lh models/sam2.1_hiera_small.pt
```

或临时设置 `SEGMENTATION_PROVIDER=mock` 做流程联调。

### 16.2 `IMAGE_EDIT_CONFIG_ERROR`

未配置 `DASHSCOPE_API_KEY`，或使用了非北京地域的 Key。万相 2.1 当前只支持华北2（北京）。

### 16.3 `IMAGE_EDIT_AUTH_ERROR`

检查：

- 是否是阿里云百炼北京地域 API Key；
- Key 是否已过期；
- `DASHSCOPE_BASE_URL` 与业务空间是否匹配。

### 16.4 `IMAGE_EDIT_FAILED` 或 `IMAGE_EDIT_CONTENT_REJECTED`

- 内容审核不通过会返回 `IMAGE_EDIT_CONTENT_REJECTED`。
- 参数或图片尺寸不合法会返回 `IMAGE_EDIT_FAILED`。
- 可查看生成步骤中的 `error_code` 和 `error_detail`。
- 不要自动重试内容审核错误。

### 16.5 `IMAGE_EDIT_TIMEOUT`

- 万相任务可能排队较慢，可增大 `IMAGE_EDIT_TIMEOUT`。
- 顺序编辑多区域会按区域累加耗时。
- 任务失败后可以 PATCH 任务后重新生成。

### 16.6 区域 `mask_ready=false`

Mask 面积异常、与原图重叠过低或分割失败。常见于：

- 鞋子被画面裁切；
- 耳环太小或分辨率不足；
- 人物 bbox 不准确；
- 物体被严重遮挡。

可以换更清晰、人物更完整的图片，或选择其他部位。

### 16.7 `replace` 提示没有物体 Mask

该区域被判断为 `missing` 或 `unknown`，没有可信的 Object Mask。应改用 `add`，或换用更清晰的图片重新分析。

### 16.8 服务管理脚本

```bash
./scripts/manage.sh doctor   # 环境、密钥、权重、PyTorch/SAM2
./scripts/manage.sh status   # PID 和健康检查
./scripts/manage.sh logs     # 查看日志
```

若提示端口被占用：

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
PORT=8001 ./scripts/manage.sh start
```

若 PID 文件残留但进程已不存在，直接重新 `start`，脚本会清理失效 PID。

### 16.9 `AI_CONNECT_IP` 与网络

如果 DeepSeek 的默认 DNS 节点连接被重置，可设置可用节点 IP：

```dotenv
AI_CONNECT_IP=120.221.61.47
```

应用会保持域名 Host 和 TLS SNI，不降低证书校验。节点 IP 可能变化，网络恢复后可删除。

---

## 17. 后续扩展

- 增加人脸/姿态关键点（MediaPipe）提升 Placement Mask 质量。
- 增加独立磁盘用量统计与定时清理任务。
- 增加多人物同一次生成。
- 增加 group / 并行编辑策略评估。
- 增加结果版本历史和重新生成。
- 增加用户鉴权、任务限流和费用统计。
- 将 SAM2 拆成独立 ML Worker，支持多进程部署。
- 接入其他图像编辑 Provider（保持 `BaseImageEditProvider` 接口）。
