# PROJECT STATUS

最后更新：2026-11-21

## 当前版本

- 应用版本：`1.3.0`
- 当前分支：`person-image-analyzer-pro1.0`
- Phase 3 提交：`c5b3752 feat: complete phase 3 image and video analysis pipeline`
- 上一回滚点：`34ad24f feat: add person image attribute analysis backend`

## 已实现功能

### 图片

- `POST /api/v1/analyze`
- JPEG / PNG / GIF / WEBP / HEIC / HEIF 内容校验
- EXIF 方向校正、尺寸压缩、RGB JPEG 归一化
- DeepSeek 全图概览 + 下装/腿部/脚部局部裁剪
- 下装 category/type、风格标签、设计特征、属性
- 丝袜、普通袜子、鞋子独立建模
- 左右腿/左右脚独立记录；无法判断时 `side=unknown`
- `image_side` 保留画面位置，不把画面左右当成人物自身左右
- `presence` 与 `visibility` 分离
- 长裙/长裤遮挡规则，不可见不判定为未穿
- 赤脚规则：有丝袜/袜子/鞋覆盖时不输出 `barefoot`
- 穿戴关系规则引擎
- 物品级 BBox，映射回原图像素
- 旧 `persons` 响应兼容
- 多人图片保留旧版结果，V2 标记 `unsupported_multi_person`

### 视频

- `POST /api/v1/analyze-video`
- `GET /api/v1/video-tasks/{task_id}`
- `GET /api/v1/video-tasks/{task_id}/result`
- `POST /api/v1/video-tasks/{task_id}/cancel`
- FFmpeg 7.1.1 LGPL 本地构建，位于 `runtime/bin`
- ffprobe 元数据校验
- 2 FPS Dense 抽帧
- OpenCV 清晰度/亮度/对比度/边缘/运动评估
- dHash 相似帧去重
- 时间分桶关键帧选择，保留首尾帧
- 单任务默认最多 12 次 AI 调用
- SQLite 任务队列和本地 Worker
- 任务进度、阶段、心跳、取消、超时、重启恢复
- 多帧融合：存在性保留、冲突记录、时间戳证据、稳定物品 ID
- 多人视频明确 `UNSUPPORTED_MULTI_PERSON`

### 工程

- Prompt 外置与版本管理
- SQLite 分析/物品/任务/Mask 元数据
- 受控本地存储、路径穿越防护、原子写入
- 分组件健康检查
- `X-Request-ID`、错误 `request_id/details`
- 请求耗时、任务 ID、调用次数、Token、分析耗时日志
- `scripts/setup.sh` / `start.sh` / `stop.sh` / `status.sh`
- README、API 文档、运行说明、FFmpeg 许可证说明

## 当前测试结果

最后运行：

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check app tests scripts
.venv/bin/python -m pip check
git diff --check
```

结果：

- pytest：**105 项全部通过**
- ruff：全部通过
- pip check：无依赖冲突
- git diff --check：通过

真实 smoke test：

- 图片：4 次调用，12,140 tokens，约 66.6 秒
- 视频：4 个分析帧，12 次调用，61,188 tokens，约 6.5 分钟
- 自动化测试不产生 API 费用

## 已知问题

- 视频 V2 单帧分析只覆盖单人；多人视频失败。
- 视频首版最多分析约 4 个关键帧，不覆盖每一帧。
- 侧向、坐姿、交叉腿、镜面自拍时左右侧经常为 `unknown`，这是保守行为。
- 长裤/长裙遮挡下的丝袜/袜子无法可靠判断。
- 尚未建立正式人工标注评估集和 P/R/F1 报告。
- 尚未实现按需 Mask（Phase 4 暂缓）。
- FFmpeg 为最小 LGPL 构建，无硬件加速。
- 当前 Phase 3 提交之后的工作区改动（如评估工具）需要单独提交。

## 下一阶段

1. Phase 5：人工标注数据集与识别质量评估。
2. Phase 6：Prompt、关键帧、API 调用和融合优化，必须通过 Phase 5 回归。
3. Phase 7：API 稳定、批量分析、日志监控、部署和交付。
4. Phase 4（按需 Mask）暂缓。

## 运行

```bash
./scripts/setup.sh
./scripts/start.sh
./scripts/status.sh
./scripts/stop.sh
```

- API：<http://127.0.0.1:8000>
- Swagger：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>
- 模型信息：<http://127.0.0.1:8000/api/v1/model/info>
