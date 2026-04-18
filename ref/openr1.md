# Open-R1 调研：可复现 R1 训练管线如何迁移到金融推理

## 1. 文献/项目摘要

Open-R1 是 Hugging Face 发起的 DeepSeek-R1 开放复现项目，目标是把 R1 风格的 reasoning model 训练流程拆成可复现组件，包括 reasoning trace 蒸馏、SFT、GRPO、评测和合成数据生成。项目地址为 https://github.com/huggingface/open-r1，官方博客为 https://huggingface.co/blog/open-r1。

它不是金融论文，但对本项目很关键：MedicalGPT 当前已经有 `training/supervised_finetuning.py`、`training/financial_grpo_training.py`、`distill/*` 和 `evaluation/evaluate_financial_benchmarks.py`，Open-R1 可以作为“工程框架参考”，帮助把金融 SFT、GRPO、数据生成和评测组织成稳定 pipeline。

## 2. 代码框架

Open-R1 的核心目录和职责：

| 模块 | 作用 | MedicalGPT 对应 |
| --- | --- | --- |
| `src/open_r1/sft.py` | reasoning traces SFT | `training/supervised_finetuning.py` |
| `src/open_r1/grpo.py` | GRPO 训练 | `training/financial_grpo_training.py` |
| `src/open_r1/generate.py` | Distilabel 合成 reasoning 数据 | `distill/distill_with_teacher.py`、`distill/run_financial_distill_pipeline.py` |
| `recipes/` | SFT/GRPO YAML recipes | `run_fingpt_v2.ipynb` 中的命令配置 |
| `scripts/run_benchmarks.py` | 多 benchmark 批量评测 | `evaluation/evaluate_financial_benchmarks.py` |
| code reward | 执行代码作为 reward | 金融 PoT/Program execution reward |

Open-R1 的工程思想可以压缩成：

```text
generate reasoning traces
-> filter / verify
-> SFT
-> GRPO with verifiable rewards
-> benchmark
-> repeat
```

## 3. 实验方法

Open-R1 的代表性实验路线：

1. 从强 reasoning teacher 蒸馏高质量 traces。
2. 用 SFT 复现 R1-Distill 类模型。
3. 对数学、代码等可验证任务做 GRPO。
4. 用 LightEval、AIME、MATH-500、GPQA、LiveCodeBench 等 benchmark 复核。
5. 对 code tasks 引入代码解释器 reward，通过沙盒执行生成代码。

对金融场景，最有参考价值的是两个点：

- SFT 与 GRPO 训练脚本解耦，用 recipe 管理参数。
- reward 不依赖主观偏好，而依赖可验证答案或代码执行结果。

## 4. 如何参考到 MedicalGPT

### 4.1 训练管线

将当前 notebook 中的手工命令逐步固化为 recipes：

```text
recipes/financial_reasoning/sft_dual.yaml
recipes/financial_reasoning/sft_etd.yaml
recipes/financial_reasoning/grpo_verifiable.yaml
recipes/financial_reasoning/eval_finqa_convfinqa.yaml
```

每个 recipe 固定 base model、train file dir、max prompt/completion length、LoRA 参数、reward 权重、output dir 和 benchmark 输出路径。

### 4.2 数据生成

Open-R1 的 `generate.py` 思路可迁移为：

```text
FinQA/ConvFinQA raw
-> teacher 生成 CoT/PoT/EoT 多候选
-> answer/program/execution verifier 过滤
-> distilled SFT / DPO / GRPO 数据
```

MedicalGPT 已有 `distill/score_distill_candidates.py`，下一步应把 PoT execution check 加入 scoring。

### 4.3 GRPO

Open-R1 的 GRPO 设计说明 MedicalGPT 不应只做格式 reward。金融 GRPO reward 推荐：

| reward | 说明 |
| --- | --- |
| answer correctness | `Normalized Answer` 与 gold 一致 |
| program execution | `Python Program` 可执行且 `ans` 正确 |
| program consistency | DSL operator 与 gold program 接近 |
| format | 输出含 `Evidence/Reasoning/Program/Answer` |
| brevity | reasoning 不过长 |

## 5. 对当前项目的落地优先级

1. 将 `run_fingpt_v2.ipynb` 中 SFT/GRPO 命令抽成 YAML/JSON manifest。
2. 在 `financial_grpo_training.py` 中重写 reward schema，兼容 `Normalized Answer` 和 PoT execution。
3. 在 distill pipeline 中加入多候选生成和 execution filtering。
4. 增加 benchmark runner，一次性比较 base/SFT/DPO/GRPO/ETD。

## 6. 风险与注意事项

- Open-R1 面向通用数学/代码，不包含金融证据定位和表格单位规范，不能直接照搬数据格式。
- Open-R1 的 code reward 针对编程题测试用例，金融 PoT reward 应执行 `ans` 并做数值容差比较。
- 金融任务的 reward precision 更依赖 `Normalized Answer`、单位归一和 program verifier。

## 7. 参考资料

- Open-R1 GitHub: https://github.com/huggingface/open-r1
- Open-R1 blog: https://huggingface.co/blog/open-r1
- DeepSeek-R1 technical report
- TRL GRPO
- MedicalGPT: `training/financial_grpo_training.py`、`distill/*`、`evaluation/evaluate_financial_benchmarks.py`
