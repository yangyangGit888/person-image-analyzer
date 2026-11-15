# AI 图片人物穿着属性识别系统

基于 FastAPI + 视觉大模型 API 的后端服务。用户通过 HTTP 上传一张人物图片，服务调用视觉模型，逐人识别人物基础信息、头部、上身、下身、腿部和脚部穿着属性，并返回统一的结构化 JSON。

当前版本只提供后端 API，不包含前端、数据库、Redis 和 Docker。图片仅在内存中处理，默认不落盘。

## 1. 技术栈

- Python 3.11+
- FastAPI + Uvicorn
- Pydantic v2 / pydantic-settings
- httpx（异步调用模型 API）
- Pillow + pillow-heif（真实图片格式校验、EXIF 方向校正、HEIC/HEIF 支持、归一化）
- python-dotenv（读取 `.env`）
- pytest（测试）

## 2. 项目结构

```text
.
├── app/
│   ├── main.py                    # FastAPI 入口、lifespan、统一异常处理
│   ├── core/
│   │   ├── config.py              # 环境变量配置与类型校验
│   │   └── exceptions.py          # 统一业务异常与错误码
│   ├── api/
│   │   ├── deps.py                # 依赖注入
│   │   └── routes/
│   │       ├── analyze.py         # POST /api/v1/analyze
│   │       ├── health.py          # GET /health
│   │       └── model.py           # GET /api/v1/model/info
│   ├── services/
│   │   ├── image_service.py       # 上传限制、真实格式校验、图片归一化
│   │   └── analysis_service.py    # Prompt 编排、JSON 校验、结果归一化
│   ├── ai/
│   │   ├── base.py                # 视觉模型统一接口
│   │   ├── openai_compatible.py   # OpenAI 兼容协议实现
│   │   ├── deepseek_provider.py   # DeepSeek 视觉模型实现
│   │   ├── parsing.py             # 模型 JSON 解析
│   │   └── factory.py             # Provider 工厂
│   ├── prompts/
│   │   └── person_analysis.py     # 人物分析 Prompt
│   └── schemas/
│       ├── analysis.py            # 分析请求/响应/三态模型
│       └── system.py              # 健康检查与模型信息模型
├── tests/
│   ├── test_health.py
│   ├── test_image_upload.py
│   ├── test_analysis.py
│   ├── test_ai_provider.py
│   ├── helpers.py
│   └── conftest.py
├── scripts/
│   └── test_client.py             # 本地图片上传测试客户端
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
└── README.md
```

## 3. 环境准备

在 macOS 上执行：

```bash
cd /Users/dongyang/Desktop/学习/python学习/test001

python3 --version          # 需要 3.11 或更高版本
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

如果只需要运行服务、不运行测试：

```bash
pip install -r requirements.txt
```

## 4. 配置 DeepSeek API Key

1. 在 <https://platform.deepseek.com/api_keys> 创建 API Key。
2. 复制环境变量模板：

```bash
cp .env.example .env
```

3. 编辑 `.env`，填入真实 Key：

```dotenv
AI_API_KEY=sk-你的真实密钥
```

`.env` 已被 `.gitignore` 忽略，不要提交到 Git，也不要把 Key 写入源码。

## 5. 配置模型名称与 API 地址

默认配置已经指向 DeepSeek 官方视觉模型：

```dotenv
AI_PROVIDER=deepseek
AI_BASE_URL=https://api.deepseek.com
AI_MODEL=deepseek-flash
AI_TIMEOUT=180
AI_MAX_RETRIES=2
AI_VISION_DETAIL=high
AI_ENABLE_THINKING=true
AI_REASONING_EFFORT=high
```

说明：

- 当前 DeepSeek 官方 API 中，`deepseek-flash` 支持图片输入；`deepseek-v4-pro` **不支持**图片输入。
- 视觉调用使用 OpenAI 兼容的 `POST /chat/completions`，图片以 base64 `data:` URL 放在 `image_url` 内容块中，并设置 `detail: high` 以保留更多细节。
- 请求同时启用 `response_format={"type": "json_object"}`，Prompt 中包含 "json" 字样和完整示例，以满足 DeepSeek JSON Output 的启用条件。
- 如果官方模型名称变更，只需修改 `AI_MODEL`。遇到未被内置识别的视觉模型时，可显式设置 `AI_MODEL_SUPPORTS_VISION=true`。
- `AI_ENABLE_THINKING=true` 会启用 DeepSeek 思考模式，准确率通常更高，但延迟和费用也会增加；可设为 `false` 提速。
- 如果本地 DNS 指向的节点连接被重置，可设置 `AI_CONNECT_IP` 直连可用节点；TLS 证书与 `Host` 头仍使用 `AI_BASE_URL` 的域名，不会降低证书校验强度。

未来接入其它厂商：

```dotenv
# OpenAI
AI_PROVIDER=openai
AI_BASE_URL=https://api.openai.com/v1
AI_MODEL=gpt-4.1

# Gemini 的 OpenAI 兼容端点（以官方最新地址为准）
AI_PROVIDER=gemini
AI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
AI_MODEL=gemini-2.5-flash
```

> 接入第三方模型前，请先确认该模型支持图片输入，并在 `AI_MODEL_SUPPORTS_VISION` 中按需声明。

## 6. 启动后端服务

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

启动后可访问：

- Swagger 文档：<http://127.0.0.1:8001/docs>
- ReDoc：<http://127.0.0.1:8000/redoc>
- 健康检查：<http://127.0.0.1:8000/health>
- 模型配置：<http://127.0.0.1:8000/api/v1/model/info>

如果未配置 API Key，服务仍可启动，`/health` 会显示 `"ai_configured": false`，分析接口会返回 `AI_CONFIG_ERROR`。

## 7. 通过 Swagger 测试

1. 打开 <http://127.0.0.1:8000/docs>。
2. 展开 `POST /api/v1/analyze`。
3. 点击 **Try it out**。
4. 在 `file` 处选择本地 JPG / JPEG / PNG / WEBP / HEIC 图片。
5. 点击 **Execute**，在 Response body 中查看结构化结果。

也可以使用 curl：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/analyze" \
  -F "file=@/path/to/photo.jpg"
```

## 8. 使用 Python 客户端上传图片

项目提供了独立客户端脚本：

```bash
source .venv/bin/activate
python scripts/test_client.py /path/to/photo.jpg
```

自定义服务地址：

```bash
python scripts/test_client.py /path/to/photo.jpg \
  --url http://127.0.0.1:8000/api/v1/analyze \
  --timeout 300
```

脚本会打印 HTTP 状态码和格式化后的 JSON；分析失败时退出码为 1。

## 9. API 说明

### 9.1 分析接口

**POST `/api/v1/analyze`**

- Content-Type：`multipart/form-data`
- 参数：`file`，图片文件
- 支持：JPG、JPEG、PNG、WEBP、HEIC/HEIF
- 按真实文件内容识别格式；iPhone 常见的 HEIC 图片即使后缀为 `.png` 也能正确处理
- 最大大小：默认 10MB（`MAX_UPLOAD_SIZE_MB`）
- 图片通过 Pillow 解析真实内容，不依赖文件后缀
- 图片会校正 EXIF 方向、转换为 RGB JPEG，长边最大 2048 像素（`IMAGE_MAX_DIMENSION`）
- 不保存上传图片

成功响应：

```json
{
  "success": true,
  "message": "分析成功",
  "data": {
    "image_info": {
      "width": 1920,
      "height": 1080,
      "person_count": 1
    },
    "persons": [
      {
        "person_id": 1,
        "position": {
          "description": "图片中央",
          "bbox": {"x": 0.35, "y": 0.1, "width": 0.3, "height": 0.8}
        },
        "visibility": {
          "full_body_visible": "unknown",
          "head_visible": "true",
          "upper_body_visible": "true",
          "legs_visible": "true",
          "feet_visible": "unknown"
        },
        "head": {
          "wearing_hat": "true",
          "hat_type": "棒球帽",
          "hat_color": "黑色",
          "wearing_glasses": "false"
        },
        "upper_body": {
          "wearing_top": "true",
          "top_type": "T恤",
          "top_color": "白色",
          "wearing_jacket": "false",
          "jacket_color": "unknown"
        },
        "lower_body": {
          "wearing_pants": "true",
          "pants_type": "牛仔裤",
          "pants_color": "蓝色",
          "wearing_skirt": "false",
          "skirt_color": "unknown"
        },
        "legs": {
          "skin_visible": "true",
          "wearing_tights": "unknown",
          "wearing_stockings": "unknown",
          "wearing_long_pants": "true",
          "wearing_shorts": "false",
          "leg_covering": "牛仔裤覆盖大腿和小腿"
        },
        "feet": {
          "wearing_socks": "unknown",
          "socks_type": "unknown",
          "socks_color": "unknown",
          "wearing_shoes": "unknown",
          "shoes_type": "unknown",
          "shoes_color": "unknown",
          "barefoot": "unknown"
        },
        "confidence": {
          "head": 0.95,
          "upper_body": 0.92,
          "lower_body": 0.9,
          "legs": 0.75,
          "feet": 0.3
        },
        "evidence": {
          "feet": "脚部被遮挡，无法判断"
        }
      }
    ]
  }
}
```

三态字段统一使用字符串：

- `"true"`：图片中明确可见为真
- `"false"`：图片中明确可见为假
- `"unknown"`：被遮挡、裁切、模糊或无法确认

文本属性无法判断时返回 `"unknown"`。`confidence` 是模型给出的参考性可见程度，不是校准概率，不能当作确定性结论。

### 9.2 健康检查

**GET `/health`**

```json
{
  "status": "ok",
  "service": "AI 图片人物穿着属性识别系统",
  "version": "1.0.0",
  "ai_configured": true,
  "timestamp": "2026-11-15T13:37:38+00:00"
}
```

### 9.3 模型配置检查

**GET `/api/v1/model/info`**

```json
{
  "success": true,
  "message": "ok",
  "data": {
    "provider": "deepseek",
    "model": "deepseek-flash",
    "base_url": "https://api.deepseek.com",
    "supports_vision": true,
    "api_key_configured": true,
    "timeout_seconds": 180.0,
    "max_retries": 2
  }
}
```

该接口不会返回 API Key。

## 10. 错误格式与错误码

所有错误使用统一结构：

```json
{
  "success": false,
  "message": "图片分析失败",
  "error": {
    "code": "AI_API_ERROR",
    "detail": "模型调用失败"
  }
}
```

| HTTP | code | 说明 |
| --- | --- | --- |
| 400 | `EMPTY_FILE` | 上传文件为空 |
| 400 | `INVALID_IMAGE` | 文件损坏或不是合法图片 |
| 413 | `IMAGE_TOO_LARGE` | 超过大小或像素限制 |
| 415 | `UNSUPPORTED_IMAGE_FORMAT` | 不是 JPG/PNG/WEBP/HEIC |
| 422 | `VALIDATION_ERROR` | 请求参数缺失或格式错误 |
| 422 | `NO_PERSON_DETECTED` | 图片中没有识别到人物 |
| 500 | `AI_CONFIG_ERROR` | 未配置 Key 或模型配置不合法 |
| 502 | `AI_AUTH_ERROR` | 模型服务鉴权失败 |
| 502 | `AI_API_ERROR` | 模型 API 调用失败 |
| 502 | `AI_INVALID_JSON` | 模型输出不是合法 JSON |
| 502 | `AI_EMPTY_RESPONSE` | 模型输出为空或被截断 |
| 502 | `AI_INVALID_RESPONSE` | 模型结果缺少必要字段 |
| 502 | `AI_MODEL_NOT_SUPPORT_VISION` | 模型不支持图片输入 |
| 503 | `AI_RATE_LIMITED` | 模型服务限流 |
| 503 | `AI_SERVICE_UNAVAILABLE` | 模型服务不可用 |
| 504 | `AI_TIMEOUT` | 模型请求超时 |
| 500 | `INTERNAL_ERROR` | 未预期的服务端异常 |

错误响应不会包含 API Key、内部环境变量或请求头。

## 11. 识别准确性设计

- Prompt 明确要求只依据实际可见像素，禁止常识推测。
- 所有三态属性只允许 `true` / `false` / `unknown`。
- 衣服、帽子、眼镜、裤子、裙子、丝袜、袜子、鞋子分别判断。
- 袜子和鞋子独立判断；看不到脚部时一律 `unknown`。
- 腿部皮肤可见不等于未穿袜子，也不等于赤脚。
- 多人物逐人独立分析，服务端强制重新编号，避免人物 ID 混用。
- 模型返回的布尔值会被归一化为三态字符串；缺失字段回退为 `unknown`。
- 模型返回非法 JSON、缺少 `persons` 数组等情况会直接报错，不猜测结果。
- 图片以高细节模式发送（`detail: high`），并在服务端校正 EXIF 方向。

## 12. 二次局部裁剪分析扩展

代码已预留扩展点：

- 配置项 `ENABLE_CROP_REFINEMENT`（默认 `false`）。
- `AnalysisService.refine_with_crops()` 方法。

推荐二期流程：

1. 首轮识别人物数量、位置与 bbox；
2. 按 bbox 裁出每个人物；
3. 对头部、上身、腿部、脚部分别调用模型；
4. 按 `person_id` 合并结果，并优先保留高置信度字段。

当前开启该开关只会记录警告并返回首轮结果，不会产生额外模型调用费用。

## 13. 运行测试

```bash
source .venv/bin/activate
pytest
```

详细输出：

```bash
pytest -v
```

测试覆盖：

- 健康检查、模型配置不泄漏 Key
- JPG/PNG/WEBP/HEIC 上传
- 伪造后缀按真实内容识别
- 损坏图片、空文件、超大文件、GIF 拒绝
- 多人物独立结果
- 三态值与缺失字段归一化
- 无人物、缺少 `persons` 字段
- 模型非法 JSON、空响应、鉴权失败、超时
- 使用 `httpx.MockTransport` 验证 DeepSeek 请求格式与重试，不消耗真实 API 费用

测试不需要真实 API Key，也不会调用真实模型。

## 14. 常见问题排查

### 14.1 分析接口返回“模型不支持图片输入”

DeepSeek 当前应使用 `deepseek-flash`。`deepseek-v4-pro` 只支持文本。检查 `.env`：

```dotenv
AI_MODEL=deepseek-flash
AI_PROVIDER=deepseek
```

如果使用自建或新模型，确认其支持视觉后设置：

```dotenv
AI_MODEL_SUPPORTS_VISION=true
```

### 14.2 返回 `AI_AUTH_ERROR`

- 检查 `AI_API_KEY` 是否正确、是否已过期。
- 确认 Key 前后没有空格或引号。
- 检查当前网络能否访问 `AI_BASE_URL`。

### 14.3 返回 `AI_TIMEOUT`

- 大图、多人图片和思考模式会增加耗时。
- 可提高 `AI_TIMEOUT`，或设置 `AI_ENABLE_THINKING=false`。
- 网络不稳定时可增加 `AI_MAX_RETRIES`。

### 14.4 返回 `AI_INVALID_JSON` 或 `AI_EMPTY_RESPONSE`

- JSON 模式要求 Prompt 中出现 "json" 并提供示例，本项目已满足。
- 输出可能因 `AI_MAX_OUTPUT_TOKENS` 太小被截断，可调大。
- 服务会自动重试部分网络错误，但空响应建议客户端重试。

### 14.5 上传后返回 `UNSUPPORTED_IMAGE_FORMAT`

- 仅支持 JPEG、PNG、WEBP、HEIC/HEIF。
- 服务按真实文件内容判断，改后缀无效。
- HEIC 由 pillow-heif 解码；GIF、BMP 等格式需先转换。

### 14.6 返回 `NO_PERSON_DETECTED`

模型认为画面中没有可识别的人物。可能是人物过小、严重遮挡、剪影或图片内容并非人物。该情况统一返回 422，不伪造结果。

### 14.7 识别结果大量 `unknown`

这是准确性优先策略的正常表现。脚部被鞋子遮挡、人物背对镜头、分辨率不足时，模型无法可靠判断袜子和鞋子，应返回 `unknown` 而不是猜测。可尝试上传更清晰、完整包含脚部的图片。

### 14.8 `ModuleNotFoundError: No module named 'app'`

请确认在项目根目录启动，并且已激活虚拟环境：

```bash
cd /Users/dongyang/Desktop/学习/python学习/test001
source .venv/bin/activate
uvicorn app.main:app --reload
```

### 14.9 端口被占用

```bash
uvicorn app.main:app --reload --port 8001
```

### 14.10 返回 `AI_API_ERROR`，日志出现 `ConnectError`

如果日志显示：

```text
模型服务连接失败: attempt=1/3 type=ConnectError
```

说明连接在 TCP/TLS 阶段被重置，通常是本地 DNS 解析到的 CDN 节点不可达，而不是 API Key 错误。可先验证：

```bash
curl -sS -o /dev/null -w '%{http_code}\n' https://api.deepseek.com/
dig @180.76.76.76 +short api.deepseek.com
```

若 curl 也出现 `Connection reset by peer`，但换一个解析结果可以访问，可在 `.env` 中指定可用节点：

```dotenv
AI_CONNECT_IP=120.221.61.47
```

设置后应用会连接该 IP，但 HTTP `Host` 和 TLS SNI 仍使用 `api.deepseek.com`，证书校验保持开启。节点 IP 可能变化，DNS 恢复正常后可删除该配置。`AI_MAX_RETRIES` 建议保持 2~4，以应对偶发网络抖动。

## 15. 后续扩展建议

- 增加网页前端：直接调用现有 `/api/v1/analyze`。
- 批量识别：在服务层复用 `AnalysisService`，增加并发控制和队列。
- 数据库：保存任务、结果和图片元数据，不要把 API Key 存入数据库。
- 结果导出：基于 `AnalyzeResponse` 模型导出 JSON / CSV / Excel。
- 局部裁剪：实现 `refine_with_crops()`，对头部和脚部做二次高细节识别。
- 接入其它视觉模型：实现 `BaseVisionProvider` 并在 `factory.py` 注册。
