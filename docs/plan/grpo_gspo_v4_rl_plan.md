# GRPO/GSPO v4 RL Plan

## Goal

Build a fresh, comparable RL protocol for the financial Program executor task. The headline question is whether GRPO or GSPO improves benchmark accuracy over the SFT2 Program model under the same data, sampling budget, LoRA setup, and evaluation protocol.

## Fixed Inputs

- Base policy: `/root/autodl-tmp/outputs/financial_reasoning_v3/sft2_program_merged`
- Tokenizer: `/root/autodl-tmp/models/qwen/Qwen2___5-7B-Instruct`
- Train data: `/root/autodl-tmp/data/financial_reasoning_cot_pot/rl/train_cot_pot_grpo_mixed.jsonl`
- Validation data: `/root/autodl-tmp/data/financial_reasoning_cot_pot/rl/valid_cot_pot_grpo_mixed.jsonl`
- Output root: `/root/autodl-tmp/outputs/financial_reasoning_rl_v4`
- Primary seed: `42`

## Training Protocol

Both GRPO and GSPO use:

- `learning_rate=1e-5`
- `beta=0.001`
- `max_steps=300`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=6`
- `num_generations=6`
- `max_completion_length=256`
- `temperature=1.10`
- `top_p=0.95`
- `loss_type=dapo`
- `multi_objective_aggregation=normalize_then_sum`
- `scale_rewards=none`
- 4-bit LoRA on `q_proj k_proj v_proj o_proj`, `r=16`, `alpha=32`, `dropout=0.05`

GSPO differs only by setting `importance_sampling_level=sequence` through `training/finqa_program_gspo.py`.

## Reward Semantics

GRPO and GSPO use the same Program reward semantics:

- executable Program reward
- execution closeness to `gold_answer`
- Program structure similarity
- Program argument coverage
- Program step count score
- exact execution match bonus
- format gate
- evidence support
- brevity regularization

GRPO keeps these as separate reward functions. GSPO keeps the existing `reward_program_core` aggregation plus auxiliary rewards. GRPO now logs the same `program/*` diagnostics as GSPO so training curves are comparable.

## Required Diagnostics

Track these in TensorBoard and `trainer_state.json`:

- `reward`
- `reward_std`
- `kl`
- `entropy`
- `frac_reward_zero_std`
- `clip_ratio/*`
- `completions/clipped_ratio`
- `program/core_score`
- `program/executable_rate`
- `program/exact_match_rate`
- `program/structure_score`
- `program/argument_coverage`
- `program/step_count_score`
- `program/has_program_rate`
- `program/unique_program_ratio`
- `program/completion_words`

Note: `completions/clipped_ratio` means generation truncation at `max_completion_length`, not policy ratio clipping.

## Execution Steps

Smoke runs:

```bash
cd /root/FinQA
MAX_STEPS=100 bash run_finqa_program_grpo_v4.sh
MAX_STEPS=100 bash run_finqa_program_gspo_v4.sh
```

Main runs:

```bash
cd /root/FinQA
bash run_finqa_program_grpo_v4.sh
bash run_finqa_program_gspo_v4.sh
```

Benchmark:

```bash
cd /root/FinQA
bash run_finqa_program_rl_benchmark_v4.sh
```

For quick benchmark checks:

```bash
CONVFINQA_MAX_SAMPLES=32 FINQA_MAX_SAMPLES=32 bash run_finqa_program_rl_benchmark_v4.sh
```

## Decision Rule

Use benchmark metrics, not training reward, as the final decision:

1. `executed_answer_accuracy`
2. `pass@1_greedy`
3. `program_execution_rate`
4. `program_accuracy`
5. `pass@4`
6. `pass@8`
7. `avg_prediction_chars`

Choose GSPO if it improves `executed_answer_accuracy` and `pass@1_greedy` without materially reducing `pass@8` or `program/unique_program_ratio`. Choose GRPO if the benchmark is tied but GRPO is simpler or more stable. If neither improves over SFT, stop extending RL steps and revisit data balance, reward weights, Program DSL coverage, and evaluation prompt alignment.

## Next Ablations

Only after the seed 42 v4 comparison is clean:

- Repeat with seeds `43` and `44`.
- Compare `num_generations=6` vs `8`.
- Compare `beta=0.0005`, `0.001`, and `0.002`.
- Try `num_iterations=2` only if KL is too low and benchmark plateaus.
