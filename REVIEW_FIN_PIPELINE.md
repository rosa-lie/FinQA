# FinGPT × MedicalGPT Pipeline Code Review (run_fingpt_min.ipynb / fin_to_sharegpt.py / fin_to_dpo_pairs.py)

## Scope
- Data preprocessing logic
- SFT + DPO pipeline logic
- Practical improvements for "industry research report assistant" scenario

## Key findings
1. Dataset schema adaptation is generally clear and reusable, but there are risks in field mapping ambiguity and task-type heuristics.
2. SFT and DPO orchestration is directionally correct (SFT->merge->DPO), but execution guardrails and evaluation are under-specified.
3. To be production-ready for financial report assistant use cases, stronger dataset governance, objective evaluation, and inference-time controllability are needed.
