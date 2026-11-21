# 评估报告

- `latest.md` / `latest.json`：当前生产 Prompt 的基线评估，17 张人工标注图片。
- `experiments/`：候选优化实验。实验结论不会自动进入生产。

## 当前基线（2026-11-21）

- 样本：17
- 失败：0
- Macro Precision：0.887
- Macro Recall：0.852
- Macro F1：0.837
- 下装 Category Accuracy：0.625
- 普通袜子 Type Accuracy：0.333
- 鞋子 Type Accuracy：0.182
- 鞋子 Group Accuracy：0.182
- Side Accuracy：N/A（标注集没有左右侧标签）
- Side Abstention Rate：0.867

## Phase 6 第一次实验

`experiments/prompt_v2_2026-11-21.md` 加强了主要人物计数、脚部可见性、鞋袜同色、袜子长度等 Prompt。

结果：

- Macro F1：0.837 → 0.841（小幅提升）
- Recall：0.852 → 0.870
- 鞋子 Recall：0.545 → 0.636
- 普通袜子 Type Accuracy：0.333 → 0.000（明显回退）
- 多人物样本仍被误判

按照“任何优化必须通过评估集回归”的规则，本次实验**未采纳**，生产 Prompt 已回退到基线。后续应拆分 Prompt 变更，逐项做 A/B 评估。
