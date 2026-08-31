# 失败路线与负结果复盘

本文保留 `README_fail_igore.md` 这个历史文件名，避免破坏已有链接。当前内容不再描述“待完成的最终方案”，而是记录本项目在 FinQA/ConvFinQA program-executor 后训练中走过的失败路线、负结果和方法边界。

早期项目把目标描述为 Fino1-style Long CoT SFT 加轻量 GRPO。这个方向有研究价值，因为金融数值推理确实需要解释证据和公式意图，但它没有抓住当前任务最容易自动验证的核心。FinQA 和 ConvFinQA 的优势不只是有自然语言答案，而是有 gold program、execution answer 和 normalized answer。只训练 CoT 会让模型更会解释，却不能保证公式可执行，也很难把 reward 设计成稳定、低噪声的自动信号。

## CoT-only 路线的问题

Fino1 和 FinCoT 更适合作为背景参考或低比例 supplement，而不适合作为当前主模型的主要监督来源。原因是它们的 reasoning path 更偏自然语言解释，格式和 FinQA DSL 不完全一致。金融数值题的核心错误通常发生在取数、单位缩放、公式方向和分母选择上，自由文本 CoT 即使写得顺畅，也可能在 executor 里完全不可执行。若把最终目标定义成泛金融 CoT 模型，评测会退化成 answer string match、LLM judge 或格式分，很难判断模型是否真正学会了金融表文计算。

当前主线因此改为 program-first。模型输出 `Evidence + Program`，系统执行 program 并比较 normalized answer。CoT 可以帮助面试讲解和错误分析，但不是 headline 评测口径。

## 早期 GRPO 为什么不稳定

直接把 base 或普通 SFT 模型送入 GRPO，在本任务上很容易失败。第一个原因是 sparse positive。金融 DSL 的合法空间很窄，模型必须同时选对证据、选对操作符、选对参数顺序、处理单位缩放并遵守输出 contract，才可能拿到正确执行奖励。早期策略生成的大量样本不可执行或答案错误，组内几乎没有正样本。

第二个原因是 wrong-executable。程序能被解析和执行不代表答案正确，例如把 percentage change 写成 `divide(current, previous)`，或者把差值方向写反，都可能产生一个合法数字。这类输出如果 reward 只奖励 parse 或 execution，就会被错误强化。

第三个原因是 zero-variance group。GRPO 的优势来自同一 prompt 下多个 completion 的相对奖励，常见形式是 $A_i=(r_i-\mu_G)/\sigma_G$ 或 Dr.GRPO 中的组内中心化变体。如果一个 group 全错、全不可执行或全对，组内方差接近零，梯度信号就会很弱。早期历史 frontier 和静态 hard bucket 容易出现这种情况，因为样本难度不是按当前策略重新估计的。

第四个原因是 proxy reward mismatch。format completeness、operator overlap、evidence keyword hit 和 brevity 可以作为辅助诊断，但它们不是最终指标。最终指标是 executor 执行 program 后是否匹配 gold normalized answer。只优化 proxy 会让模型学到“看起来像 program”的输出，而不是稳定答对。

## 历史 frontier 的局限

历史 hard examples 不等于当前策略的可学习样本。一个样本曾经被旧模型答错，不代表当前 RS-SFT 模型仍然在边界上；反过来，一个当前模型 greedy 答错但 pass@8 中存在正确程序的样本，才更适合 GRPO。v34r23 之后改用 current-policy frontier acquisition，筛选标准是 greedy wrong、采样候选中至少一个 executor-correct program，并且最好存在 executable-wrong hard negative。这个筛选把 RL 预算集中在模型已经接近但还没把 greedy 行为校准好的区域。

## DPO 为什么没有成为主线

DPO 适合有明确 chosen/rejected 偏好对的任务，但 FinQA/ConvFinQA 的主要监督信号是可执行 program 的答案正确性。早期 DPO 或蒸馏式偏好数据没有稳定超过 SFT2 的 pass@k 结果，且 chosen/rejected 往往混入格式、表达风格、长度和答案正确性的多重差异。对于当前任务，这会让目标变得不够干净。项目最终选择 RS-SFT 先吃掉高质量 executor-correct 样本，再用 GRPO 做 current-policy frontier 上的小步校准。

## 当前负结果如何写进结论

当前 full-1237 结果是 Base 0.272433、SFT2 0.578820、RS-SFT 0.603072、v34r23 GRPO 0.606306、v34r24 Dr.GRPO 0.607114。这个序列说明主要能力来自 Program SFT 和 retention-aware RS-SFT。GRPO 在强基线后有小幅正收益，但不是从 27.24% 到 60.71% 的主要来源。

v34r24 Dr.GRPO checkpoint-50 相比 RS-SFT 多 5 题，相比 v34r23 GRPO 多 1 题。它可以被描述为窄幅 late-stage calibration，不能被描述为已经充分解决 length bias、difficulty bias 或 reward hacking。后续如果要证明 Dr.GRPO 的系统性价值，需要更多 checkpoint、更多 seed、分桶稳定性分析，以及对 changed predictions 的定性审计。

这份失败复盘的面试表达应当是，项目并不是简单“用了 GRPO 所以提升很大”，而是先把任务改造成可验证 program generation，再用 SFT 和 RS-SFT 建立强 executor baseline，最后只在当前策略可学习的 frontier 上做 RL 校准。这个过程体现的是问题定义、数据筛选、reward 设计和评测口径的迭代能力。
