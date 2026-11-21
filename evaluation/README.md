# 识别质量评估集

## 数据来源

- 原始文档：`Doc1.docx`，共 17 页，对应 17 张图片。
- 本目录中的图片已按文档出现顺序编号为 `img_001` 至 `img_017`。
- 原始 Doc1.docx 未被修改；编号通过提取文件、`dataset/manifest.json` 和联系表体现。
- 标注文件：`annotations/ground_truth.jsonl`，每行一个 `sample_id`。

出于隐私考虑，以下内容默认不提交 Git：

```text
evaluation/dataset/images/
evaluation/dataset/contact_sheet.jpg
evaluation/annotations/
evaluation/predictions/
```

`dataset/manifest.json` 记录页码、文件名称和图片尺寸，可提交。

## 目录

```text
evaluation/
├── dataset/
│   ├── images/img_001.png ... img_017.png
│   ├── manifest.json
│   └── contact_sheet.jpg
├── annotations/ground_truth.jsonl
├── predictions/predictions.jsonl
├── reports/latest.md
├── reports/latest.json
├── metrics.py
└── run_evaluation.py
```

## 运行

先确保 `.env` 已配置 DeepSeek Key：

```bash
.venv/bin/python -m evaluation.run_evaluation --concurrency 2
```

只跑部分样本：

```bash
.venv/bin/python -m evaluation.run_evaluation --ids img_001,img_002
```

重新运行：

```bash
.venv/bin/python -m evaluation.run_evaluation --force
```

脚本支持断点续跑：已有 `sample_id` 会跳过；失败样本默认重试。

## 指标

- 分目标 Precision / Recall / F1
- Visibility Accuracy
- Category Accuracy：下装类别、普通袜子类型、鞋子类型/类别组
- Abstention Rate：`uncertain` 视为主动放弃判断
- Not Visible Rate：`not_visible` 是可观察性判断，不直接算识别错误
- Side Abstention Rate：物品级 `side=unknown` 比例
- Side Accuracy：当前标注集没有左右侧标签，报告为 N/A

## 重要原则

- 模型保守返回 `unknown` / `uncertain` 不等于识别错误。
- 交叉腿、镜面自拍、局部裁剪导致左右侧无法确定时，`unknown` 可能是正确行为。
- 长裙/长裤遮挡时，腿脚物品 `not_visible` 不应被计为“未穿”。
- 当前 17 张样本是小规模初始评估集，不代表完整生产准确率。
- 优化 Prompt 或调用策略后，必须重新运行本评估集并对比报告。
