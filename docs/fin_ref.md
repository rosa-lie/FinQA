distill finqa

# [**Unlocking Data Value in Finance: A Study on Distillation and Difficulty-Aware Training**](https://arxiv.org/pdf/2603.07223)

本文要点：区分sft&rl；多源数据集处理；蒸馏数据高质量。


> **在金融垂直领域里，真正决定效果的，不只是模型大小或训练技巧，而是 post-training 数据的“质量、难度、可验证性”**。高质量 CoT 蒸馏更适合 SFT 阶段，而高难且可验证的数据更适合 RL 阶段。

> 近期金融领域 LLM 研究表明，在垂直金融任务中，模型性能的关键不只在于模型规模或训练算法，而在于 post-training 数据的质量、难度与可验证性。特别地，高质量 CoT 蒸馏数据更适合用于 SFT 阶段，以帮助模型建立稳定的金融推理结构；而高难度且可验证的数据更适合用于后续 RL 阶段，以提升模型在复杂金融数值推理任务中的泛化与鲁棒性。因此，本项目采用“高质量蒸馏 SFT + 可验证偏好优化/RL”的训练路线，以构建面向金融对话数值推理的小模型。([Hugging Face][1])

> **需要蒸馏，但不要让 teacher 直接替做题；而要让 teacher 在 gold 约束下，把“正确答案”转写成“高质量、可验证、可学习的金融推理监督信号”。**

---

**金融领域模型提升，核心不是盲目堆训练，而是把不同“价值类型”的数据，放到正确的训练阶段里。**
* **SFT 阶段**：重点吃“**高质量、可解释、结构清晰的蒸馏 CoT 数据**”
* **RL 阶段**：重点吃“**高难度、可验证、能算对错的金融推理数据**”
* 如果数据只是“多”，但不“干净/可验证/有难度分层”，提升会很有限，甚至会误导模型。([Hugging Face][1])

---

作者认为：**通用大模型虽然很强，但在金融场景里经常不够可靠**，主要有三个痛点：
1. **金融术语密集**
2. **数值推理要求高**
3. **事实错误容忍度极低**  
所以金融模型不能只会“像样地说”，而要能：读表格 结合文本 做多步数值推理 输出稳定、可信、可检查的结果

> **论文的方法主线：不是换模型，而是“重新设计训练数据”**. 
> 论文的贡献主要不是提出一个很花哨的新训练算法，而是提出：**数据应该分层治理**

A. **ODA-Fin-SFT-318k** 用于 **SFT**
* 多阶段 distillation（多轮蒸馏）
* verification（验证）
* 目标是得到**高质量 CoT supervision**

也就是说，它不是随便拿 teacher 模型输出当标签，而是要经过筛选和验证，尽量让训练信号“可靠、结构化、能学到思维方式”。([Hugging Face][1])

B. **ODA-Fin-RL-12k** 用于 **RL**
* hard-but-verifiable（难，但可验证）
* 强调 difficulty-aware（难度感知）
* 强调 reward precision（奖励可精确判断）

这说明他们非常明确地区分：
* **SFT 要“可模仿”**
* **RL 要“可判分”**

> 这个 distinction 非常重要，而很多项目恰恰会把这两类数据混在一起，结果两边都学不好。

---

## **高质量 CoT 蒸馏是 SFT 的“地基”**
> **高质量 Chain-of-Thought distillation establishes a robust foundation during SFT**。

SFT 阶段，真正该学的不是“答案文本”，而是“金融推理过程的组织方式”。
* 如何从题目中抽取关键信息
* 如何识别表格字段
* 如何决定先算什么后算什么
* 如何把推理过程组织成稳定格式. 
这比“最后 answer 是多少”更关键。

> 如果 DeepSeek-R1 直接回答 FinQA，很多时候它会说“信息不足无法回答”，那这种蒸馏是不是没效果？

这篇论文给出的答案基本是：**对，直接这么蒸馏，效果会很差。**  
因为这时候 teacher 输出的不是 正确推理轨迹 稳定结构 可学习的 reasoning pattern，而只是拒答 模糊回答 错误推理 无法泛化的噪声文本。  
这种数据喂给学生模型，只会让它学会：“不会就拒答” “不会时胡乱生成模板话术” “对金融题形成错误先验”。  
这不是要的金融 reasoning 能力。

## **蒸馏不是“让 teacher 替答题”，而是“让 teacher 帮组织推理监督信号”**

> 不应该让 R1 直接“作答”，而应该让它基于 gold_response 去“生成更高质量 reasoning / CoT 轨迹”。

因为对 FinQA / ConvFinQA 这种任务来说，**gold answer 是可靠锚点**。  
而 teacher 模型真正的价值，是* **把正确答案“展开成可学习的 reasoning form”**  
也就是 teacher 更适合做“解释器 / 组织器 / 重写器”，而不是“最终求解器”。


## 蒸馏范式：**Gold-guided Distillation**

不是prompt → teacher → response
而是**prompt + gold answer (+ 若有程序/中间量更好)** → teacher 生成 → “结构清晰、步骤合理、格式统一”的 reasoning trace
* 解释为什么答案是这个
* 补出可读的中间步骤
* 规范表达
* 对齐最终 gold

如果把 teacher 的任务改成“已知正确答案是 X，请基于材料写出一段严谨、简洁、金融推理风格的 reasoning”，它的成功率会比“裸答题”高很多。这就更接近论文里说的：**高质量、验证过的 CoT supervision**，而不是“teacher 原生生成文本的随机产物”。

## **RL 阶段不要吃“普通题”，而要吃“难但能判分的题”**

1）Difficulty-aware（难度感知）
* 容易出错
* 需要多步推理
* 需要选择正确计算路径
* 能区分强模型和弱模型的题

## 2）Verifiable（可验证）
* ConvFinQA：答案可比对
* 表格推理：中间程序/运算可检查
* 数值题：可做 tolerance-based reward

> **金融 reasoning 这种任务，最有价值的 RL 数据，不是“主观偏好”，而是“客观可验证”**

## **“数据价值”比“数据数量”更重要**：**Unlocking Data Value**。

也就是说，数据不是越多越好，而是要看它有没有“训练价值”。

**数据分类：**  
第一类：低价值数据：teacher 拒答；teacher 幻觉；reasoning 很空泛；答案和 gold 不一致；过程和结论矛盾；模板化废话太多。这些数据哪怕量大，也会污染 SFT。   
第二类：中价值数据：最终答对了，reasoning 基本合理，但过程比较啰嗦、格式不稳定。这类可以通过清洗、压缩、统一模板后再用。   
第三类：高价值数据：think&answer都符合期待（程序运算正确，表哥字段引用正确，中间变量清楚，最终答案正确，格式稳定可模仿，能直接作为 reasoning supervision）

## 蒸馏失败最常见的 4 个原因

1）teacher 本身在任务上不稳定：R1 在某些金融题上会“信息不足”，这说明 teacher 对本项目任务分布并不完全适配。   
2）teacher 生成没有被验证：如果不做 verification，错误 CoT 会直接污染训练。
3）蒸馏的是“语言表面风格”，不是“可学习推理结构”：比如学到很多“让我们一步一步分析” “根据题意可知” “因此答案是”；但没有真正学到 数值抽取 变量绑定 运算路径。那就只是“会装 reasoning”。   
4）把不适合 SFT 的样本硬塞进 SFT（超长噪声 reasoning 模糊/不确定的题 需要 RL 才能提升的难题）这会拖累 SFT。

## 路线：**原始金融推理数据** → **高质量蒸馏与清洗** → **分阶段 SFT** → **轻量 preference / verifiable RL** → **benchmark 评估**

### 1. 数据集

阶段 A：SFT 数据构建（最关键）：构建“金融 reasoning 地基数据”。  
不要再用“裸 teacher 作答蒸馏”，改成 **Gold-guided Distillation**

```text
你是一名严谨的金融推理助手。
请阅读题目材料、问题和标准答案。
本项目任务不是重新猜答案，而是：
1. 根据材料找出与问题相关的信息；
2. 用简洁、正确、可验证的步骤解释为什么标准答案成立；
3. 不要编造材料中没有的信息；
4. 最终结论必须与标准答案完全一致。
```

输入：

* context / table / text
* question
* **gold answer**
* （如果有）gold program / operator chain

输出：

* structured reasoning
* final answer

对蒸馏结果做过滤 **distill verifier**。
* 最终答案 == gold
* 输出不含“信息不足”“无法判断”等拒答模板
* reasoning 中至少包含 2~4 个有效步骤
* reasoning 长度不要过长（防止废话）
* 数值/单位与 gold 对齐

* 若有 program，则 reasoning 与 program 大体一致
* 表格字段引用合理
* 没有明显 hallucination

最终宁缺毋滥：**保留 40%~70% 的高质量蒸馏样本 丢弃 30%~60% 的垃圾蒸馏样本**

### 2. 分阶段SFT

**SFT-1：结构化单轮金融数值推理**
- 数据来源： FinQA TaTQA（可选） 清洗后的单轮蒸馏数据
- 目标： 学会表文联合 学会基本数值推理 学会稳定格式

**SFT-2：多轮金融对话推理**
- 数据来源： ConvFinQA 多轮 reasoning 数据
- 目标： 学会跨轮引用 学会在对话中延续计算状态 学会 follow-up reasoning

> “分阶段训练”方向是对的，这篇论文的结论也支持这种“先打基础，再上高难”的思路。因为单轮数值推理和多轮对话推理，本质难点不完全一样。如果一开始就混着训，小模型很容易 两边都学不牢 对话格式学会了，但数值能力没打实 或数值能力有一点，但对话追踪不稳。

---

### RL 

**后训练阶段要上“难且可验证”的样本，而不是泛泛 preference 数据。** DPO / GRPO 不应该做成“聊天偏好对齐”，而应该 **金融 reasoning 偏好与可验证优化**

DPO 数据构造方式：
- 正样本（chosen）（正确答案 推理简洁 结构清晰 引用材料准确）
- 负样本（rejected）（算错 漏步骤 错字段 错单位 推理跳步太大 空泛模板话术）

完全可以做 rule-based reward：“verifiable”
* **Answer reward**：最终答案对不对
* **Format reward**：是否符合结构模板
* **Consistency reward**：过程与答案是否一致
* **Program reward**（若有）：与 gold program 是否接近
* **Numerical reward**：数值误差是否在阈值内

---

## 局限

- 它更像“数据工程研究”，不是强方法创新；
- 默认 teacher / verifier 本身足够强；
- 它适合“金融领域整体增强”，但是“金融数值推理专项”（表格 数值 多轮上下文 公式运算）（应该做得比它更“窄、更硬核”。
- “长链推理是否真的学到了”：即使蒸馏了 CoT，也不代表模型真的内化了推理。所以后面评估时一定要区分：会不会答对，是不是靠真正 reasoning 答对。这就需要设计更细的 benchmark 分析。


---

[1]: https://huggingface.co/papers/2603.07223?utm_source=chatgpt.com "Paper page - Unlocking Data Value in Finance: A Study on Distillation and Difficulty-Aware Training"

---

# [Progressive Knowledge Distillation and Numerical Reasoning Enhancement for Financial Report Question Answering](https://www.mdpi.com/2079-9292/14/23/4653)