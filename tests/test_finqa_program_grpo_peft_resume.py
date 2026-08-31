from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "training" / "finqa_program_grpo.py"


def test_grpo_script_accepts_existing_peft_adapter_path():
    source = SOURCE.read_text(encoding="utf-8")

    assert "PeftModel" in source
    assert "peft_path" in source
    assert "PeftModel.from_pretrained" in source
    assert "is_trainable=True" in source
