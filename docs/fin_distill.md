# 金融数值推理数据蒸馏计划

## 1. 目标

本计划用于在现有 `FinQA / ConvFinQA -> SFT v2 -> DPO -> GRPO` 框架上补一条可控的数据蒸馏链路，用更强 teacher 模型为金融数值推理任务生成更高质量的监督信号。

蒸馏目标不是单纯“扩充更多样本”，而是生成和筛选更稳定的以下能力：

- 结构化回答能力：`问题分析 / 关键证据 / 推理程序 / 最终答案`
- 数值答案正确率
- 程序推理一致性
- 多轮对话压缩后的有效推理能力
- 后续 DPO / GRPO 可用的偏好和 reward 信号

第一阶段只做 `FinQA` 和 `ConvFinQA`，不把 `Fineval`、`FIQA_QA` 等开放类数据混入蒸馏主链路。

## 2. 设计原则

### 2.1 与训练格式保持一致

teacher 输出必须和当前训练格式一致，避免蒸馏数据与现有 SFT / DPO / benchmark 分布不一致。

目标格式固定为：

```text
问题分析：...
关键证据：
- ...
推理程序：...
最终答案：...
```

### 2.2 从 raw record 出发，不从旧 SFT target 出发

蒸馏输入应直接来自原始数据记录，经统一 processor 处理 prompt；不要从旧版 `*_clean_strict.jsonl` 的答案侧反向改写，否则会把旧 target 的偏差一并继承进去。

### 2.3 自动校验优先于人工挑选

蒸馏链路必须把自动验证放在中心位置。只有通过结构、答案、程序、长度等检查的样本，才能进入训练集。

### 2.4 先蒸 SFT，再蒸 DPO，再考虑 GRPO

如果蒸馏后的 SFT 数据都不能让模型至少不弱于 `base`，继续堆 DPO / GRPO 只会放大偏差。

## 3. 蒸馏对象

### 3.1 SFT 蒸馏

目标：为每条题目生成一个高质量、结构稳定、证据更自然的 teacher answer。

输出文件建议：

- `distill_sft_finqa.jsonl`
- `distill_sft_convfinqa.jsonl`
- `distill_sft_joint.jsonl`

每条记录至少包含：

```json
{
  "prompt": "...",
  "response": "问题分析：...
关键证据：...
推理程序：...
最终答案：...",
  "gold_answer": "127.40",
  "gold_program": "divide(637, const_5)",
  "task_name": "finqa_train",
  "source_dataset": "FinQA",
  "record_id": "V/2008/page_17.pdf-1",
  "teacher_name": "...",
  "validation": {
    "answer_correct": true,
    "program_consistent": true,
    "structured": true
  }
}
```

### 3.2 DPO 蒸馏

目标：对同一题保留多个 teacher 候选，并构造 chosen / rejected 对。

输出文件建议：

- `distill_dpo_finqa.jsonl`
- `distill_dpo_convfinqa.jsonl`
- `distill_dpo_joint.jsonl`

chosen / rejected 的来源建议：

- chosen：结构完整、答案正确、程序一致的最佳候选
- rejected：
  - 答案错但结构对
  - 程序错但答案表面对
  - 证据冗长或仍有 JSON 痕迹
  - 缺失 `最终答案` 或结构不完整

### 3.3 Verifier 蒸馏

目标：保留所有候选与校验信息，作为后续分析和 reward 调优依据。

输出文件建议：

- `distill_audit_finqa.jsonl`
- `distill_audit_convfinqa.jsonl`

## 4. 蒸馏输入构造

### 4.1 统一复用现有 processor

蒸馏输入必须直接复用当前 family processor：

- `financial_data_processors/families/finqa.py`
- `financial_data_processors/families/convfinqa_turn.py`

这样可以保证：

- 训练集和蒸馏集 prompt 分布一致
- benchmark 与训练 prompt 口径一致
- `关键证据` 已经走 `gold_ind/gold_inds` 摘要逻辑
- `ConvFinQA` 历史问题压缩口径保持一致

### 4.2 teacher 输入内容

teacher 输入保留：

- 当前问题
- 压缩后的上下文
- 截断后的表格
- 必要的历史问题

teacher 输入不包含：

- gold answer
- gold program
- 旧版 target

### 4.3 多候选采样

每题建议生成 `3-5` 个候选：

- 1 个低温保守候选
- 2-4 个中温探索候选

示例策略：

- `temperature=0.0`：追求稳定结构
- `temperature=0.4~0.7`：追求多样性和可选 rejected

## 5. 自动校验规则

每条候选至少执行以下校验。

### 5.1 结构校验

检查是否包含：

- `问题分析：`
- `关键证据：`
- `推理程序：`
- `最终答案：`

失败样本：

- 直接丢弃，或仅保留到 audit 文件

### 5.2 数值答案校验

优先做数值归一化比较：

- 纯数值
- 百分比
- 货币符号
- 千分位逗号

需要特别处理：

- `12%` 与 `0.12`
- `$94` 与 `94`
- `127.4` 与 `127.40`

### 5.3 程序一致性校验

对 `FinQA / ConvFinQA`：

- 优先比较规范化后的 program 字符串
- 次级比较 operator 序列
- 保留“答案对但程序不一致”的样本，供 DPO rejected 或 audit 使用

### 5.4 证据质量校验

检查 `关键证据`：

- 是否仍包含明显 JSON 痕迹，如 `{"text_1":`、`{"table_1":`
- 是否过长
- 是否重复

### 5.5 长度校验

统计：

- prompt token 长度
- response token 长度
- 总长度是否超过训练预算

## 6. 数据产物

建议把蒸馏过程拆成三个阶段文件。

### 6.1 输入文件

由 raw 数据 + processor 生成：

- `distill_input_finqa.jsonl`
- `distill_input_convfinqa.jsonl`

### 6.2 候选文件

每题多个候选，带 teacher 原始输出：

- `distill_candidates_finqa.jsonl`
- `distill_candidates_convfinqa.jsonl`

### 6.3 过滤后文件

正式训练使用：

- `distill_sft_*.jsonl`
- `distill_dpo_*.jsonl`
- `distill_audit_*.jsonl`

## 7. 推荐脚本拆分

建议新增三类脚本。

### 7.1 `build_financial_distill_dataset.py`

职责：

- 从 raw record 生成蒸馏输入
- 复用现有 processor
- 输出 prompt、gold answer、gold program、metadata

### 7.2 `distill_with_teacher.py`

职责：

- 调用 teacher 模型生成多候选
- 保存原始输出
- 记录采样参数、teacher 名称、时间戳

### 7.3 `score_distill_candidates.py`

职责：

- 结构校验
- 数值答案校验
- 程序一致性校验
- 证据和长度审计
- 生成 SFT / DPO / audit 三类输出

## 8. notebook 接入计划

在 `run_fingpt_min.ipynb` 中新增 `Distill` section，顺序建议如下：

1. 构造蒸馏输入集
2. 运行 teacher 生成候选
3. 执行候选打分与过滤
4. 输出蒸馏统计报告
5. 基于蒸馏数据训练 `SFT distill`
6. 基于同题多候选训练 `DPO distill`
7. 用 quick benchmark 比较：
   - `base`
   - `sft_v2`
   - `sft_distill`
   - `sft_distill + dpo_distill`

## 9. 评估策略

蒸馏效果不能只看通过率，要固定看 benchmark。

至少比较：

- `FinQA answer_accuracy`
- `ConvFinQA answer_accuracy`
- `program_accuracy`
- `structured_response_coverage`
- `final_answer_coverage`

推荐流程：

1. 先在小样本 quick eval 上验证方向
2. 再在完整 dev benchmark 上复核
3. 如果 `sft_distill` 仍弱于 `base`，优先回查 teacher 输出和过滤规则

## 10. 分阶段落地顺序

### Phase 1

只做 `FinQA` 蒸馏：

- 单一 teacher
- 每题 3 个候选
- 先生成 `distill_sft_finqa.jsonl`
- benchmark 验证 `base vs sft_v2 vs sft_distill`

### Phase 2

加入 `ConvFinQA`：

- 引入历史问题压缩后的 prompt
- 单独分析多轮上下文对 teacher 输出的影响

### Phase 3

构建 `DPO distill`：

- 每题保留 chosen / rejected
- 对比 `sft_distill` 与 `sft_distill + dpo_distill`

### Phase 4

最后才考虑 `GRPO`：

- 在蒸馏后模型已经不弱于 `base` 的前提下
- 再补 verifier / reward 驱动优化

## 0. 当前进度

### 已完成

- 新增 `build_financial_distill_dataset.py`
  - 从 `FinQA / ConvFinQA` raw 数据复用现有 family processor 构造蒸馏输入
  - 默认对 `ConvFinQA` 复用最终轮去重口径
- 新增 `distill_with_teacher.py`
  - 支持 `openai`、`gold`、`copy_gold_final` 三种 backend
  - 支持多候选采样、温度调度、断点续跑
- 新增 `score_distill_candidates.py`
  - 对 teacher 候选做结构、答案、程序、JSON 痕迹校验
  - 直接产出训练可用的 `SFT` 与 `DPO` 数据文件
- 已完成 smoke test
  - `build -> generate -> score` 三段链路已跑通
  - `gold` backend 可产出 `distill_sft.jsonl`
  - 注入坏候选后，`distill_dpo.jsonl` 选择逻辑验证通过

### 当前限制

- 还没有把蒸馏 section 接入 `run_fingpt_min.ipynb`
- 还没有真正调用外部 teacher API 生成金融 teacher 候选
- `score_distill_candidates.py` 当前的 program 一致性仍是字符串级比较，后续可加强为 operator 级或 AST 级比较

## 11. 当前建议

基于当前实验结果，最合理的下一步不是直接扩大训练规模，而是：

1. 先把 `SFT v2` 数据重建完成并验证不再弱于 `base`
2. 在此基础上只对 `FinQA` 做第一版蒸馏
3. 用 benchmark 决定是否继续扩到 `ConvFinQA` 与 `DPO distill`

这样可以把变量控制在最小范围内，避免在当前 SFT 基线尚不稳定时，把更多 teacher 偏差放进训练链路。
