# tests/test_configurator.py

from pathlib import Path

import yaml

import finetune_forge.agents.configurator as cfg_mod
from finetune_forge.agents.configurator import run_configurator
from finetune_forge.backends.llamafactory import build_llamafactory_yaml, get_template
from finetune_forge.schemas.state import ModelConfig, TrainingConfig, DatasetInfo


def test_get_template_known_families():
    assert get_template("meta-llama/Llama-3.1-8B-Instruct") == "llama3"
    assert get_template("Qwen/Qwen3-7B") == "qwen3"
    assert get_template("microsoft/Phi-3.5-mini-instruct") == "phi3"
    assert get_template("mystery/model") == "default"


def test_build_yaml_lora(tmp_path):
    mc = ModelConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        model_size_b=8.0,
        quantization="none",
        training_method="lora",
        target_modules=["q_proj", "v_proj"],
    )
    tc = TrainingConfig(output_dir=str(tmp_path / "out"))
    path = build_llamafactory_yaml(mc, tc, dataset_path=str(tmp_path / "processed_dataset.json"))

    config = yaml.safe_load(Path(path).read_text())
    assert config["finetuning_type"] == "lora"
    assert config["stage"] == "sft"
    assert config["lora_target"] == "q_proj,v_proj"
    assert config["template"] == "llama3"
    assert "quantization_bit" not in config


def test_build_yaml_qlora_adds_quantization(tmp_path):
    mc = ModelConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        model_size_b=8.0,
        quantization="4bit",
        training_method="qlora",
        target_modules=["q_proj"],
    )
    tc = TrainingConfig(output_dir=str(tmp_path / "out"))
    path = build_llamafactory_yaml(mc, tc, dataset_path=str(tmp_path / "processed_dataset.json"))
    config = yaml.safe_load(Path(path).read_text())
    assert config["quantization_bit"] == 4
    assert config["quantization_method"] == "bitsandbytes"


def test_run_configurator(monkeypatch, base_state, tmp_path):
    base_state["model_config"] = ModelConfig(
        model_name="microsoft/Phi-3.5-mini-instruct",
        model_size_b=3.8,
        training_method="lora",
        target_modules=["q_proj", "v_proj"],
    )
    base_state["dataset_info"] = DatasetInfo(
        raw_path="x.json",
        format="sft",
        processed_path=str(tmp_path / "processed_dataset.json"),
        num_examples=3,
    )

    monkeypatch.setattr(
        cfg_mod,
        "call_llm",
        lambda **kwargs: {
            "num_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "max_seq_length": 2048,
            "fp16": False,
            "bf16": True,
            "logging_steps": 10,
            "save_strategy": "epoch",
        },
    )
    # Keep output inside tmp_path.
    monkeypatch.chdir(tmp_path)

    out = run_configurator(base_state)
    assert out["error"] is None
    assert out["training_config"] is not None
    assert Path(out["llamafactory_yaml_path"]).exists()


def test_run_configurator_no_model_config(base_state):
    base_state["model_config"] = None
    out = run_configurator(base_state)
    assert "no model_config" in out["error"]
