# FinGPT 数据 + MedicalGPT 训练框架：最小可跑流程

目标：
- 数据：使用 FinGPT 提供的金融任务数据
- 方法：使用 MedicalGPT 的多阶段训练方法（本最小流程先跑 SFT + LoRA 合并）

## 1. FinGPT 仓库
- GitHub: https://github.com/AI4Finance-Foundation/FinGPT

## 2. 新增脚本
- `fin_to_sharegpt.py`：将 FinGPT 风格数据转换为 MedicalGPT SFT 格式（ShareGPT conversations）
- `fin_to_dpo_pairs.py`：将 FinGPT 风格数据转换为 MedicalGPT DPO 偏好对格式
- `run_fingpt_min.ipynb`：最小可跑 Notebook（加载 -> 转换 -> SFT -> 合并 LoRA）

## 3. 一键最小流程
```bash
# 推荐：直接打开并逐个执行 Notebook cells
# jupyter lab run_fingpt_min.ipynb
#
# 或在支持 notebook 的 IDE 中打开：run_fingpt_min.ipynb
```

参数修改方式：
- 直接在 `run_fingpt_min.ipynb` 的「配置参数」cell 里修改 `BASE_MODEL / FIN_DATASET / FIN_SPLIT / OUT_DIR` 等变量

## 4. 分步执行（更可控）

### 4.1 下载 FinGPT 数据到本地 jsonl
```bash
python - <<'PY'
import json
from pathlib import Path
from datasets import load_dataset

ds = load_dataset('FinGPT/fingpt-sentiment-train', split='train')
out = Path('data/fingpt_min/raw/fingpt_raw.jsonl')
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', encoding='utf-8') as f:
    for row in ds:
        f.write(json.dumps(dict(row), ensure_ascii=False) + '\n')
print('saved:', len(ds), out)
PY
```

### 4.2 转 SFT 格式
```bash
python fin_to_sharegpt.py \
  --source_file data/fingpt_min/raw/fingpt_raw.jsonl \
  --output_file data/fingpt_min/sft/fingpt_sft_sharegpt.jsonl
```

### 4.3 转 DPO 偏好对格式
```bash
python fin_to_dpo_pairs.py \
  --source_file data/fingpt_min/raw/fingpt_raw.jsonl \
  --output_file data/fingpt_min/dpo/fingpt_dpo_pairs.jsonl
```

### 4.4 用 MedicalGPT 跑 SFT（LoRA）
```bash
python supervised_finetuning.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --tokenizer_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train_file_dir data/fingpt_min/sft \
  --validation_split_percentage 1 \
  --do_train \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --max_steps 50 \
  --logging_steps 5 \
  --save_steps 50 \
  --model_max_length 512 \
  --output_dir outputs/fingpt_sft_lora
```

### 4.5 合并 LoRA
```bash
python merge_peft_adapter.py \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --tokenizer_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --lora_model outputs/fingpt_sft_lora \
  --output_dir outputs/fingpt_sft_merged
```

## 5. 说明
- `fin_to_dpo_pairs.py` 优先复用已有 `response_chosen/response_rejected` 字段；若原始数据是单答案样本，则自动生成一个弱 `rejected` 作为最小可跑方案。
- 若要获得高质量 DPO/RM 效果，建议使用人工标注或高质量自动构造的偏好对。
