# 金融强化学习训练路线

本文档说明 `run_fingpt_v2_rl.ipynb` 的设计目标、数据来源、GRPO 框架选择、reward 口径和 benchmark 验收标准。它承接已经训练好的 `sft2_dual_merged`，不重新训练 SFT。

## 定位

当前金融推理主链已经完成：

```text
FinQA strict-A -> SFT-1
ConvFinQA turn-level strict-A + FinQA replay -> SFT-2
```

RL 阶段从以下模型继续：

```text
/root/autodl-tmp/outputs/financial_reasoning_v2/sft2_dual_merged
```

RL 的目标不是让回答更像偏好样本，而是让模型在可验证金融数值推理上更稳定地答对：

- `Normalized Answer` 更准确
- `Program` 更接近 gold program
- 输出格式稳定
- 推理简洁，不膨胀成长篇 CoT
- 不遗忘 ConvFinQA 多轮 follow-up 能力

此前 DPO 的 pass@k 结果没有超过 `sft2_merged_passk`，因此第一版 RL 不以 DPO adapter 为起点。

## Notebook 入口

独立 notebook：

```text
/root/MedicalGPT/run_fingpt_v2_rl.ipynb
```

它与 `run_fingpt_v2.ipynb` 解耦：

- 不复用旧的 `training/financial_grpo_training.py`
- 不依赖旧的中文 `问题分析 / 推理程序 / 最终答案` reward schema
- 直接使用 TRL `GRPOTrainer`
- reward 函数在 notebook 内显式定义，便于快速调试

## 框架选择

第一版使用 Hugging Face TRL 的 `GRPOTrainer`。

| 框架 | 适用场景 | 当前定位 |
| --- | --- | --- |
| TRL `GRPOTrainer` | 单机、LoRA/PEFT、自定义 Python reward、快速研究 | 默认 |
| verl | FSDP/vLLM/SGLang、多组件 rollout 和 reward 管理 | 后续扩展 |
| OpenRLHF | Ray + vLLM + remote reward server + 大规模 RLHF | 后续扩展 |

先用 TRL 验证数据和 reward 是否有效。如果 benchmark 有收益，再考虑迁移到 verl 或 OpenRLHF。

## 数据来源

第一版混合两类数据。

### 1. 核心可验证数据

来源：

```text
/root/autodl-tmp/data/financial_reasoning_v2/clean/train_sft1_dual_strict.jsonl
/root/autodl-tmp/data/financial_reasoning_v2/clean/train_sft2_convfinqa_turn_dual_strict.jsonl
/root/autodl-tmp/data/financial_reasoning_v2/clean/train_sft2_finqa_replay_dual.jsonl
```

用途：

- 保留 FinQA / ConvFinQA 的 gold `Program`
- 保留 `Normalized Answer`
- 作为 program-verifiable reward 的主数据

统一成 GRPO 行：

```json
{
  "prompt": "...",
  "answer": "...",
  "gold_program": "...",
  "source_dataset": "finqa|convfinqa_turn",
  "task_type": "program_verifiable",
  "reward_profile": "program_numeric"
}
```

core 内部默认比例：

```text
ConvFinQA : FinQA = 2 : 1
```

这个比例延续 SFT-2 的能力来源：以 ConvFinQA turn-level 为主，同时通过 FinQA replay 保持单轮表文推理。

### 2. FinCoT RL 数据

来源：

```python
load_dataset("TheFinAI/FinCoT", split="RL")
```

字段映射：

| FinCoT 字段 | RL notebook 字段 | 用途 |
| --- | --- | --- |
| `Question` | `prompt` | 训练 prompt |
| `Answer` | `answer` | answer reward |
| `Reasoning_process` | `reference_reasoning` | 参考推理，不直接作为 gold program |
| `Final_response` | `reference_response` | 参考回答 |
| `Negative_reasoning_process` | `negative_reasoning` | 分析或后续 rejection-sampling |
| `Negative_response` | `negative_response` | negative-response avoidance 启发式 |

统一成 GRPO 行：

```json
{
  "prompt": "...",
  "answer": "...",
  "gold_program": "",
  "reference_reasoning": "...",
  "reference_response": "...",
  "negative_reasoning": "...",
  "negative_response": "...",
  "source_dataset": "fincot_rl",
  "task_type": "cot_verifiable",
  "reward_profile": "answer_format"
}
```

重要约束：

- FinCoT 不强行伪造 `Program`
- 允许 `Program: N/A`
- 不把 FinCoT negative 字段直接转成 DPO
- 第一版只用 answer、format、brevity 和 negative-response avoidance 作为轻量 reward

### 3. Distill 数据

已有 distill 产物：

```text
/root/autodl-tmp/data/financial_reasoning/finqa_distill_r1_program_2048/distill_sft.jsonl
/root/autodl-tmp/data/financial_reasoning/finqa_distill_r1_program_2048/distill_summary.csv
```

第一版只审计，不默认混入 GRPO。后续加入前必须检查：

- answer 可解析
- program 可解析或明确标记 `Program: N/A`
- 不与 FinQA / ConvFinQA eval/test prompt 高重叠
- 输出不过长

## 混合比例

默认采样：

```python
MAX_CORE_RL_ROWS = 3500
MAX_FINCOT_RL_ROWS = 1500
VALID_RATIO = 0.05
SEED = 42
```

约等于：

```text
FinQA / ConvFinQA core program data: 70%
FinCoT RL data: 30%
```

如果发现 ConvFinQA 能力下降，降低 FinCoT 比例到 10%-20%。

## 输出格式

RL 阶段继续使用 v2 英文 schema：

```text
Evidence:
- ...

Reasoning:
...

Program: ...

Answer: ...
Normalized Answer: ...
```

规则：

- FinQA / ConvFinQA 必须输出 `Program`
- FinCoT 可以输出 `Program: N/A`
- `Reasoning` 保持短，不鼓励长篇 CoT
- `Normalized Answer` 是主要数值评测字段

## Reward 设计

Notebook 使用多个 custom reward functions。每条样本通过 `reward_profile` 选择不同权重。

### program_numeric

用于 FinQA / ConvFinQA。

| reward | 权重 | 说明 |
| --- | ---: | --- |
| normalized answer correctness | 0.50 | `Normalized Answer` 与 gold answer 数值匹配 |
| program parse / operator consistency | 0.20 | `Program` 中的 operator 与 gold program 接近 |
| program-answer consistency | 0.10 | program 存在且最终答案可解析 |
| format completeness | 0.10 | 包含核心 anchors |
| evidence grounding | 0.05 | Evidence 中出现 prompt 关键数字 |
| brevity | 0.05 | 输出长度不过度膨胀 |

### answer_format

用于 FinCoT RL。

| reward | 权重 | 说明 |
| --- | ---: | --- |
| answer correctness | 0.60 | 答案数值或文本与 gold answer 匹配 |
| format completeness | 0.15 | 包含 `Evidence / Answer / Normalized Answer` |
| reasoning relevance / concise length | 0.15 | 输出长度合理 |
| negative-response avoidance | 0.10 | 不复述 FinCoT negative response |

FinCoT 样本不因为 `Program: N/A` 扣 program reward。

## 训练参数

默认 GRPO 参数：

```python
GRPO_NUM_GENERATIONS = 4
MAX_PROMPT_LENGTH = 1536
MAX_COMPLETION_LENGTH = 384
LEARNING_RATE = 5e-6
BETA = 0.001
MAX_STEPS = 300
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
USE_VLLM = False
```

训练开关：

```python
RUN_GRPO_SMOKE = False
RUN_FULL_GRPO = False
```

必须先跑 `RUN_GRPO_SMOKE=True` 的 1 step smoke test，再启动完整训练。

## Benchmark

训练后用统一 pass@k 设置比较：

```text
--finqa_max_samples 8
--convfinqa_max_samples 8
--pass_k 1,4,8
--num_samples_per_example 8
--sample_temperature 0.7
--sample_top_p 0.95
--sample_seed 42
```

主要对照：

- `base_passk`
- `sft2_merged_passk`
- `dpo_passk`
- `grpo_fincot_passk`

验收标准：

| 指标 | 要求 |
| --- | --- |
| macro answer accuracy | 不低于 `sft2_dual_merged` |
| program accuracy | 不明显下降，最多允许约 3pp 波动 |
| pass@4 / pass@8 | 不低于 `sft2_dual_merged` |
| FinQA accuracy | 优先观察是否提升 |
| ConvFinQA accuracy | 不应遗忘 |
| avg prediction chars | 不应明显膨胀 |

不要只看训练期 reward。最终结论必须来自 benchmark。

## Smoke Test 顺序

推荐按以下顺序运行 notebook：

1. import 依赖
2. 检查 `sft2_dual_merged` 和数据路径
3. 构造 core RL 数据
4. 加载 FinCoT RL split
5. 写出 `train_grpo_mixed.jsonl`、`valid_grpo_mixed.jsonl`、`smoke_grpo_mixed.jsonl`
6. 跑 handcrafted reward smoke test
7. 设置 `RUN_GRPO_SMOKE=True` 跑 1 step GRPO
8. 确认 adapter 可保存
9. 设置 `RUN_FULL_GRPO=True` 跑完整训练
10. 跑 pass@k benchmark
11. 查看结果分析表和 correct/wrong flip analysis

## 后续扩展

### Difficulty-aware GRPO

参考 Cao et al. 的 hard-but-verifiable 思路：

1. 用 `sft2_dual_merged` 对训练候选题采样 4-8 次
2. 计算每题 pass rate
3. 选择 `0.2 <= pass_rate <= 0.7`
4. 过滤不可验证、答案过长和证据不清样本
5. 进入 GRPO

### FinanceReasoning

FinanceReasoning 带 `python_solution`，适合后续作为：

- 外部 benchmark
- Python Program execution reward 数据
- by difficulty / by operator count 分桶评估

第一版不直接混入训练，避免数据 schema 和去污染问题影响主实验判断。

### verl / OpenRLHF

如果 TRL 单机 GRPO 有收益，再考虑迁移：

- verl：适合 FSDP/vLLM/SGLang、多组件 rollout 和 reward manager
- OpenRLHF：适合 Ray + vLLM、remote reward server 和更大规模 actor/reference/reward 部署

迁移时保持 reward contract 不变：

```text
prompt
answer
gold_program
reward_profile
reference_reasoning / negative_response
```

不要在验证 reward 和数据有效前过早迁移框架。
