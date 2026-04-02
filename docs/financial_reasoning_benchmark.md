# 金融推理 benchmark 评估流程

目标：
- 比较 `基座模型`
- 比较 `SFT1` 对英文金融数值推理的提升
- 比较 `SFT1 + SFT2` 对中文金融知识与泛化的追加提升
- 检查 `SFT2` 是否破坏了 `SFT1` 已学到的推理能力

## 1. benchmark 设计

本项目当前采用三类评测任务：ConvFinQA test 看英文金融数值推理和表格理解，重点衡量 SFT1 增益；fingpt-fineval test 看中文金融知识和考试型推理，重点衡量 SFT2 增益；CFLUE 小规模外部任务集看中文金融泛化，避免只在训练分布内变强。打分维度上，ConvFinQA 以答案准确率为主、程序一致率为辅；Fineval/CFLUE 以选择题答案准确率为主。结果解读时重点看三组差值：sft1 - base、sft2 - sft1、sft2 - base，同时检查 sft2 是否在 ConvFinQA 上出现遗忘。

### 1.1 ConvFinQA test
- 目标：评估英文金融数值推理、表格理解、多轮上下文跟踪
- 用途：最直接反映 `SFT1` 增益
- 数据来源：本地 `ConvFinQA test_turn.json`
- 主指标：`answer_accuracy`
- 辅指标：`program_accuracy`

说明：
- `answer_accuracy` 优先按数值容差匹配
- 若无法解析为数值，则回退到规范化字符串匹配
- `program_accuracy` 通过比较生成结果中的 `推理程序` 字段与 gold program 的规范化字符串得到

### 1.2 fingpt-fineval test
- 目标：评估中文金融知识、考试型推理、选项判断
- 用途：最直接反映 `SFT2` 增益
- 数据来源：默认通过 Hugging Face `FinGPT/fingpt-fineval` 的 `test` split 加载
- 主指标：`answer_accuracy`

说明：
- 当前实现按生成式评测统一口径执行
- 优先从 `最终答案` 字段抽取选项字母 `A-F`
- 若无法抽取标准选项，则回退到规范化字符串匹配

### 1.3 CFLUE 小规模外部任务集
- 目标：评估中文金融泛化，而不是只看训练分布内收益
- 用途：验证 `SFT1 + SFT2` 是否在外部金融考试/知识任务上继续提升
- 数据来源：用户本地准备的若干 `json/jsonl` 文件
- 主指标：每个子任务的 `answer_accuracy`

推荐先选 2 到 4 个客观题子任务：
- 银行从业
- 证券从业
- 基金从业
- 保险相关知识题

原因：
- 这类任务与 `fingpt-fineval` 同属金融知识推理，但来源不同
- 当前脚本已对选择题生成式评测做了稳定支持
- 可以避免在第一版 benchmark 中同时引入阅读理解、生成摘要等任务专属打分器

## 2. 模型对比方式

统一评估 3 个模型：
- `base`：基座模型
- `sft1`：完成 `SFT1` 并 merge 后的模型
- `sft2`：完成 `SFT1 + SFT2` 并 merge 后的模型

建议统一解码参数：
- `temperature=0.0`
- `max_new_tokens=256`
- `repetition_penalty=1.0`

这样做的目的：
- 降低采样噪声，方便直接对比 checkpoint
- 让 benchmark 更像稳定测试，而不是自由生成展示

## 3. 结果判读

建议重点看以下三个差值：
- `Delta(sft1 - base)`：看 `SFT1` 是否显著提升 `ConvFinQA`
- `Delta(sft2 - sft1)`：看 `SFT2` 是否显著提升 `fingpt-fineval` 与 `CFLUE`
- `Delta(sft2 - base)`：看完整两阶段训练的总体收益

同时重点排查：
- `sft2` 在 `ConvFinQA` 上是否回退
- `sft2` 在 `fingpt-fineval` 上是否提升但 `CFLUE` 无提升

若出现上述情况，通常意味着：
- replay 比例偏低
- `SFT2` 过度贴合考试题分布
- 指令模板或输出格式在阶段切换中发生漂移

## 4. 评测脚本

新增脚本：`evaluate_financial_benchmarks.py`

功能：
- 一次性评估多个模型
- 支持 `ConvFinQA test`
- 支持 `fingpt-fineval test`
- 支持本地 `CFLUE` 小任务集
- 导出逐条预测与汇总结果

输出文件：
- `benchmark_manifest.json`
- `{model_name}_predictions.jsonl`
- `{model_name}_summary.jsonl`
- `benchmark_summary.json`
- `benchmark_summary.csv`

## 5. 命令示例

```bash
python evaluate_financial_benchmarks.py \
  --tokenizer_path /root/autodl-tmp/models/qwen/Qwen2___5-7B-Instruct \
  --model_entry base=/root/autodl-tmp/models/qwen/Qwen2___5-7B-Instruct \
  --model_entry sft1=/root/autodl-tmp/outputs/financial_reasoning/sft1_merged \
  --model_entry sft2=/root/autodl-tmp/outputs/financial_reasoning/sft2_merged \
  --convfinqa_test_file /root/autodl-tmp/data/financial_reasoning/raw/convfinqa_turn/test_turn.json \
  --fineval_dataset_name FinGPT/fingpt-fineval \
  --fineval_split test \
  --cflue_task_file bank=/root/autodl-tmp/data/financial_reasoning/raw/cflue/bank.jsonl \
  --cflue_task_file securities=/root/autodl-tmp/data/financial_reasoning/raw/cflue/securities.jsonl \
  --output_dir /root/autodl-tmp/outputs/financial_reasoning/benchmarks/sft2_compare
```

## 6. notebook 集成方式

`run_fingpt_min.ipynb` 已补充 benchmark 配置与执行单元。

执行顺序：
1. 跑完 `SFT-2`
2. merge `SFT-2 LoRA`
3. 执行 benchmark 单元
4. 自动评估 `base / sft1 / sft2`
5. 结果保存到 `OUTPUT_ROOT / benchmarks / sft2_compare`

## 7. 注意事项

- `ConvFinQA test` 不要参与任何训练或清洗回流。
- 若 `fingpt-fiqa_qa` 已参与 `SFT2`，不要再把它当最终 benchmark。
- `CFLUE` 建议先用选择题子集，等主流程稳定后再扩展到阅读理解或生成任务。
- 若显存紧张，可在评估时加 `--load_in_8bit` 或 `--load_in_4bit`。
- 若只做快速回归测试，可设置：
  - `--convfinqa_max_samples`
  - `--fineval_max_samples`
  - `--cflue_max_samples_per_task`
