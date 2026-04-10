# 项目架构整理

## 当前结论

当前仓库已经具备按领域拆分的主干结构，主体实现集中在功能子目录中，根目录当前只保留训练编排 shell 和实验 notebook 入口。

本次整理后：

1. `pipelines/scripts/` 已删除
2. `pipelines/notebooks/` 中的 notebook 已移动到仓库根目录
3. shell 编排和实验 notebook 默认都从仓库根目录启动

这意味着当前目录组织的原则已经比较明确：**功能实现放在子目录，根目录只保留 shell 编排和 notebook。**

## 目录分层

### 1. 训练层

- `training/`
- 职责：SFT、DPO、GRPO、PPO、ORPO、RM、PT 等训练实现
- 代表模块：`training/supervised_finetuning.py`、`training/dpo_training.py`

### 2. 推理与服务层

- `serving/`
- 职责：离线推理、Gradio、FastAPI、OpenAI-compatible API、ChatPDF
- 代表模块：`serving/inference.py`、`serving/openai_api.py`

### 3. 数据处理层

- `data/`
- 职责：清洗、审计、过滤、格式校验
- 代表模块：`data/clean_sharegpt_dataset.py`、`data/filter_sharegpt_by_audit.py`

### 4. 金融数据路由层

- `financial_data_processors/`
- 职责：按数据族做解析、统一转换成 SFT / DPO 所需格式
- 入口：`financial_data_router.py`
- 特点：这是目前金融子项目里最接近可复用领域模块的部分

### 5. 蒸馏流水线层

- `distill/`
- 职责：蒸馏数据构造、teacher 生成、多候选打分、SFT/DPO/audit 导出
- 入口：`distill/run_financial_distill_pipeline.py`
- 特点：已呈现 pipeline 组织方式，是金融 reasoning 主链路的重要中枢

### 6. 评测层

- `evaluation/`
- 职责：量化评测与金融 benchmark 评测
- 代表模块：`evaluation/eval_quantize.py`、`evaluation/evaluate_financial_benchmarks.py`

### 7. 工具层

- `tooling/`
- 职责：tokenizer 构建、adapter 合并、量化等辅助工具

### 8. 根目录编排层

- 职责：直接可运行的 shell 脚本与实验 notebook
- 代表文件：`run_sft.sh`、`run_dpo.sh`、`run_fingpt_min.ipynb`、`run_training_dpo_pipeline.ipynb`

### 9. 配置层

- `configs/`
- 职责：deepspeed / zero 配置

### 10. 文档层

- `docs/`
- 职责：数据、蒸馏、训练参数、FAQ 与架构说明

## 路径约定

当前建议的使用方式如下：

```bash
python -m training.supervised_finetuning ...
python -m training.dpo_training ...
python -m serving.inference ...
python -m data.clean_sharegpt_dataset ...
python financial_data_router.py ...
python -m distill.run_financial_distill_pipeline ...
sh run_sft.sh
```

Notebook 统一位于仓库根目录：

- `run_fingpt_min.ipynb`
- `run_fingpt_distill.ipynb`
- `run_training_dpo_pipeline.ipynb`
- `run_training_ppo_pipeline.ipynb`

## 本次已完成

- 删除了 `pipelines/scripts/`
- 将 `pipelines/notebooks/` 中的 notebook 收敛到仓库根目录
- 修正了仓库内与旧路径相关的文档引用

## 一句话原则

**以后新增或修改功能时，优先维护功能子目录中的真实实现；根目录只保留 shell 编排和 notebook。**
