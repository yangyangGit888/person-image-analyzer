# 识别质量评估报告

- 样本数：17
- 失败样本：0
- Macro Precision：0.887
- Macro Recall：0.852
- Macro F1：0.837

## 分目标指标

| 目标 | N | GT 正例 | TP | FP | FN | Precision | Recall | F1 | Visibility Acc | Abstention | Not Visible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lower_garment | 17 | 16 | 15 | 1 | 1 | 0.938 | 0.938 | 0.938 | 0.706 | 0.0% | 0.0% |
| pantyhose | 17 | 1 | 1 | 1 | 0 | 0.500 | 1.000 | 0.667 | 0.588 | 17.6% | 0.0% |
| ordinary_socks | 17 | 9 | 7 | 0 | 2 | 1.000 | 0.778 | 0.875 | 0.765 | 5.9% | 23.5% |
| footwear | 17 | 11 | 6 | 0 | 5 | 1.000 | 0.545 | 0.706 | 0.706 | 11.8% | 11.8% |
| bare_feet | 17 | 3 | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.765 | 0.0% | 11.8% |

## 类别与侧向

- 下装 Category Accuracy：0.625 (10/16)
- 普通袜子 Type Accuracy：0.333 (3/9)
- 鞋子 Type Accuracy：0.182 (2/11)
- 鞋子 Group Accuracy：0.182 (2/11)
- Side Accuracy：N/A（当前标注集没有左右侧标签）
- Side Abstention Rate：0.867 (26/30)

## 样本明细

| 样本 | lower | pantyhose | socks | footwear | bare feet |
|---|---|---|---|---|---|
| img_001 | detected→detected / clear→partial | detected→detected / clear→clear | not_visible→not_visible / not_visible→not_visible | detected→detected / clear→clear | not_detected→not_detected / clear→clear |
| img_002 | detected→detected / partial→partial | not_detected→uncertain / clear→clear | detected→not_visible / clear→not_visible | not_visible→not_visible / not_visible→not_visible | not_detected→not_visible / clear→partial |
| img_003 | detected→detected / clear→clear | not_detected→uncertain / clear→clear | detected→detected / clear→clear | detected→not_detected / clear→clear | not_detected→not_detected / clear→clear |
| img_004 | detected→detected / partial→partial | not_detected→uncertain / clear→partial | detected→detected / clear→clear | detected→detected / clear→clear | not_detected→not_detected / clear→clear |
| img_005 | detected→detected / partial→partial | not_detected→not_detected / clear→clear | detected→detected / clear→clear | detected→uncertain / clear→clear | not_detected→unknown / clear→clear |
| img_006 | detected→detected / partial→partial | not_detected→not_detected / partial→clear | not_detected→not_detected / clear→clear | not_detected→not_detected / clear→clear | detected→detected / clear→clear |
| img_007 | not_visible→detected / not_visible→partial | not_detected→not_detected / partial→clear | detected→detected / clear→clear | not_visible→not_detected / not_visible→clear | not_detected→not_detected / clear→clear |
| img_008 | detected→detected / partial→partial | not_detected→not_detected / clear→partial | not_detected→not_detected / clear→clear | not_detected→not_detected / clear→clear | detected→detected / clear→clear |
| img_009 | detected→detected / clear→clear | not_detected→not_detected / partial→clear | not_visible→uncertain / not_visible→partial | detected→uncertain / clear→partial | not_visible→unknown / not_visible→partial |
| img_010 | detected→detected / partial→partial | not_detected→not_detected / clear→clear | not_detected→not_detected / clear→clear | not_detected→not_detected / clear→clear | detected→detected / clear→clear |
| img_011 | detected→detected / partial→partial | not_detected→not_detected / clear→clear | not_visible→not_detected / partial→clear | detected→detected / clear→clear | not_visible→not_detected / partial→clear |
| img_012 | detected→detected / clear→clear | not_detected→not_detected / clear→clear | detected→detected / clear→clear | detected→detected / clear→clear | not_detected→not_detected / clear→clear |
| img_013 | detected→detected / clear→clear | not_detected→not_detected / clear→clear | detected→detected / clear→clear | detected→detected / clear→clear | not_detected→not_detected / clear→clear |
| img_014 | detected→unknown / partial→not_visible | not_visible→unknown / not_visible→not_visible | not_visible→unknown / not_visible→not_visible | detected→unknown / partial→not_visible | not_visible→unknown / not_visible→not_visible |
| img_015 | detected→detected / partial→clear | not_visible→not_detected / not_visible→clear | not_visible→not_visible / not_visible→not_visible | detected→not_visible / partial→not_visible | not_visible→not_visible / not_visible→not_visible |
| img_016 | detected→detected / clear→clear | not_detected→detected / clear→clear | detected→not_visible / partial→not_visible | detected→detected / clear→clear | not_detected→not_detected / clear→partial |
| img_017 | detected→detected / partial→clear | not_detected→not_detected / clear→partial | detected→detected / clear→clear | not_visible→not_detected / not_visible→clear | not_detected→not_detected / clear→clear |
