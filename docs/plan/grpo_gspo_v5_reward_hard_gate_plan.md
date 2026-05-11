# GRPO v5 Hard-Gated Reward Plan

## Background

The current GRPO-300 run underperformed the SFT v3 Program model on the quick benchmark. The main regression is on FinQA: execution accuracy dropped from `0.875` to `0.5`, program execution rate dropped from `1.0` to `0.75`, and average prediction length grew from `135.625` to `412.5` characters. This indicates reward hacking and objective mismatch rather than under-training.

## Objective

Train a conservative GRPO policy that preserves SFT's executable Program ability while only rewarding short, valid, execution-correct completions. The training target is `executed_answer_accuracy` and `program_execution_rate`, not longer reasoning text.

## Reward Changes

- Missing, `Program: N/A`, parse-failing, or execution-failing Program gets a fixed reward of `-1.0`.
- Executable but answer-wrong Program is capped at `0.2`.
- Executable and answer-correct Program starts at `1.0`, with only small structure, argument, step-count, and brevity bonuses.
- Format hacking is penalized through forbidden anchors, multiple `Program:` sections, and extra text after the final Program.

## Training Protocol

- Base model: `/root/autodl-tmp/outputs/financial_reasoning_v3/sft2_program_merged`
- Output root: `/root/autodl-tmp/outputs/financial_reasoning_rl_v5`
- `learning_rate=5e-6`
- `beta=0.002`
- `max_steps=200`
- `max_completion_length=192`
- `temperature=1.0`
- `top_p=0.95`
- `loss_type=dapo`
- `scale_rewards=none`
- `save_steps=50`

Run smoke first:

```bash
cd /root/FinQA
MAX_STEPS=50 bash run_finqa_program_grpo_v5.sh
```

Run main only if smoke does not degrade benchmark:

```bash
cd /root/FinQA
bash run_finqa_program_grpo_v5.sh
```

## Evaluation

Use the v5 benchmark script to evaluate SFT v3 and any available GRPO checkpoints with the same prompt/output protocol:

```bash
cd /root/FinQA
bash run_finqa_program_grpo_v5_benchmark.sh
```

Acceptance criteria:

- `program_execution_rate >= SFT`
- FinQA `executed_answer_accuracy` must not regress.
- Macro `avg_prediction_chars` should not exceed `1.5x` SFT.
- If checkpoint-50 already regresses, stop and revise data/reward before longer training.

## Next Step if v5 Still Regresses

Do not increase steps. Build a balanced or hard-example dataset: FinQA/ConvFinQA 1:1, with oversampling for `greedy wrong but pass@8 correct` examples.
