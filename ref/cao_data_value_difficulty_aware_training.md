# Unlocking Data Value in Finance 调研：蒸馏与难度感知训练

## 1. 文献摘要

论文：`Cao 等 - 2026 - Unlocking data value in finance a study on distillation and difficulty-aware training.pdf`

该文提出一个 data-centric 的金融后训练路线，核心观点是：在金融垂直领域，性能主要由 post-training 数据的质量、难度和可验证性决定，而不是单纯由模型规模决定。

论文构建两个数据集：

- `ODA-Fin-SFT-318k`：通过多阶段蒸馏和验证构造的高质量 CoT SFT 数据。
- `ODA-Fin-RL-12k`：面向 RL 的 hard-but-verifiable 数据，强调 reward precision 和 task diversity。

公开信息显示，作者使用标准 SFT 和 RL/GRPO 管线训练 `ODA-Fin-RL-8B`，并在 9 个金融相关 benchmark 上超过同规模开源金融 LLM。论文页为 https://huggingface.co/papers/2603.07223。

## 2. 代码框架

该论文的可复用框架：

```text
raw financial datasets
-> semantic deduplication
-> multi-stage CoT distillation
-> verifier filtering
-> ODA-Fin-SFT-318k
-> SFT
-> generate model pass/fail statistics
-> select hard-but-verifiable samples
-> ODA-Fin-RL-12k
-> GRPO/RL
-> nine-benchmark evaluation
```

关键设计：

| 组件 | 作用 |
| --- | --- |
| semantic dedup | 去除重复和近重复样本 |
| CoT distillation | 用强模型生成 reasoning traces |
| verification | 短答案用规则/验证器，长答案用 judge |
| hard sample mining | 选择 SFT 模型失败率较高的样本 |
| verifiability filter | 保留答案短、可自动判分的任务 |
| GRPO | 用精确 reward 推高泛化 |

## 3. 实验方法

论文实验重点：

1. 对 raw financial data 做蒸馏和验证。
2. 比较 raw data、partial distillation、full distillation 的效果。
3. 构建 hard-but-verifiable RL 数据。
4. 比较从 base model RL 和从 SFT model RL 的差异。
5. 在多个金融 benchmark 上评估，包括金融理解、情感、数值推理等。

公开摘要中的关键结论：

- 高质量 CoT 蒸馏适合 SFT 阶段。
- 困难且可验证的数据适合 RL 阶段。
- 直接用 raw/noisy 数据训练可能伤害强基座。
- RL 必须从强 SFT 初始化开始，直接从 base RL 可能退化。

## 4. 如何参考到 MedicalGPT

### 4.1 数据质量优先

当前 MedicalGPT 已有 strict-A、audit、normalized answer parse 等机制，应进一步吸收 ODA 的 data-centric 思路：

```text
raw FinQA/ConvFinQA
-> strict-A program verification
-> ETD distillation
-> execution verification
-> SFT data
```

不要把所有外部 CoT 数据直接混入训练，必须先通过 answer check、program execution check、evidence check、length check 和 duplication check。

### 4.2 难度感知 RL

构建 GRPO 数据时，不应随机抽样。推荐：

1. 用当前 SFT 模型对候选训练题采样 4-8 次。
2. 计算 pass rate。
3. 选择 pass rate 在 `0.2-0.7` 的题作为 hard-but-learnable。
4. 过滤掉答案太长或不可自动判分的题。
5. 进入 GRPO。

### 4.3 与 PoT 结合

ODA 主要强调 CoT 蒸馏；MedicalGPT 应升级为：

```text
high-quality CoT
+ executable PoT
+ EoT verification
+ difficulty-aware GRPO
```

也就是说，SFT 阶段学结构化 reasoning，RL 阶段只吃 hard-but-verifiable 的数值/程序题。

## 5. 对当前项目的实验建议

| 实验 | 目的 |
| --- | --- |
| raw strict-A only | 当前主链 baseline |
| ETD distilled SFT | 验证高质量蒸馏收益 |
| raw + external CoT | 验证噪声/负迁移风险 |
| hard random GRPO | baseline |
| difficulty-aware GRPO | 验证难度感知收益 |
| pass-rate bucket eval | 分析模型在哪些难度区间提升 |

建议新增文件：

```text
/root/autodl-tmp/data/financial_reasoning_v2/grpo/hard_verifiable_train.jsonl
/root/autodl-tmp/data/financial_reasoning_v2/reports/difficulty_buckets.json
```

## 6. 风险与注意事项

- 难题不等于好 RL 数据；必须同时可验证。
- 答案过长的开放题 reward 噪声大，不适合第一版 GRPO。
- 外部通用数学 CoT 可能造成负迁移，金融任务应优先 domain-aligned data。
- Raw data 对强模型可能是负收益，必须先做过滤和蒸馏。

## 7. 参考资料

- Cao et al. 2026. Unlocking Data Value in Finance: A Study on Distillation and Difficulty-Aware Training.
- Hugging Face paper page: https://huggingface.co/papers/2603.07223
- ODA-Fin-SFT-318k
- ODA-Fin-RL-12k
- MedicalGPT strict-A / dual answer / GRPO pipeline
