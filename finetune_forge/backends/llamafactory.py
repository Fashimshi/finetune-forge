# finetune_forge/backends/llamafactory.py
"""LlamaFactory backend: YAML config builder + subprocess runner.

This module owns everything that knows about LlamaFactory's on-disk config
format and CLI, so the agents (configurator/executor) stay backend-agnostic.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import yaml

from finetune_forge.schemas.state import ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)


# Map our TrainingMethod -> LlamaFactory training stage.
_STAGE_MAP = {
    "lora": "sft",
    "qlora": "sft",
    "full_ft": "sft",
    "dpo": "dpo",
    "reward_modeling": "rm",
}


def get_template(model_name: str) -> str:
    """Returns LlamaFactory chat template identifier for known models."""
    name = model_name.lower()
    if "llama-3" in name:
        return "llama3"
    if "llama-2" in name:
        return "llama2"
    if "qwen3" in name:
        return "qwen3"
    if "qwen2" in name:
        return "qwen"
    if "phi-3" in name or "phi3" in name:
        return "phi3"
    if "phi-4" in name or "phi4" in name:
        return "phi4"
    if "mistral" in name:
        return "mistral"
    if "gemma" in name:
        return "gemma"
    if "deepseek" in name:
        return "deepseek"
    return "default"


def write_dataset_info(
    dataset_path: str,
    stage: str,
    dataset_name: str = "custom_dataset",
) -> str:
    """Register the processed dataset in LlamaFactory's ``dataset_info.json``.

    LlamaFactory resolves a dataset by looking up ``dataset_name`` in
    ``<dataset_dir>/dataset_info.json``; without this entry training aborts with
    a "dataset not found" error. Returns the path to the written file.
    """
    data_path = Path(dataset_path)
    dataset_dir = data_path.parent

    if stage == "dpo":
        # Paired-preference (ranking) format produced by the data processor.
        entry = {
            "file_name": data_path.name,
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        }
    elif stage == "pretrain" or stage == "pt":
        entry = {"file_name": data_path.name, "columns": {"prompt": "text"}}
    else:  # sft / rm default to the alpaca-style mapping.
        entry = {
            "file_name": data_path.name,
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }

    info_path = dataset_dir / "dataset_info.json"
    existing: dict = {}
    if info_path.exists():
        try:
            existing = json.loads(info_path.read_text())
        except json.JSONDecodeError:
            logger.warning("Existing dataset_info.json is invalid; overwriting.")
    existing[dataset_name] = entry
    info_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    return str(info_path)


def build_llamafactory_yaml(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    dataset_path: str,
    dataset_name: str = "custom_dataset",
) -> str:
    """Build a LlamaFactory-compatible train-args YAML file and return its path.

    Also writes the companion ``dataset_info.json`` so LlamaFactory can resolve
    the processed dataset.

    Reference: https://github.com/hiyouga/LlamaFactory/blob/main/examples/
    """
    method = model_config.training_method
    stage = _STAGE_MAP.get(method, "sft")

    # Register the dataset so LlamaFactory can find it at train time.
    write_dataset_info(dataset_path, stage, dataset_name)

    config: dict = {
        "model_name_or_path": model_config.model_name,
        "stage": stage,
        "do_train": True,
        "finetuning_type": "lora" if method in ("lora", "qlora") else "full",
        "dataset": dataset_name,
        "dataset_dir": str(Path(dataset_path).parent),
        "template": get_template(model_config.model_name),
        "cutoff_len": training_config.max_seq_length,
        "max_samples": 100000,
        "overwrite_cache": True,
        "preprocessing_num_workers": training_config.dataloader_num_workers,

        # Output
        "output_dir": training_config.output_dir,
        "logging_steps": training_config.logging_steps,
        "save_steps": 500,
        "plot_loss": True,
        "overwrite_output_dir": True,

        # Training
        "per_device_train_batch_size": training_config.per_device_train_batch_size,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
        "learning_rate": training_config.learning_rate,
        "num_train_epochs": training_config.num_epochs,
        "lr_scheduler_type": training_config.lr_scheduler_type,
        "warmup_ratio": training_config.warmup_ratio,
        "fp16": training_config.fp16,
        "bf16": training_config.bf16,
        "ddp_timeout": 180000000,

        # Eval
        "val_size": 0.1,
        "per_device_eval_batch_size": 1,
        "eval_strategy": "steps",
        "eval_steps": 500,
    }

    # LoRA-specific settings
    if method in ("lora", "qlora"):
        config.update({
            "lora_rank": model_config.lora_rank,
            "lora_alpha": model_config.lora_alpha,
            "lora_dropout": model_config.lora_dropout,
            "lora_target": ",".join(model_config.target_modules),
        })

    # QLoRA quantization
    if method == "qlora":
        config.update({
            "quantization_bit": 4 if model_config.quantization == "4bit" else 8,
            "quantization_method": "bitsandbytes",
        })

    yaml_path = Path(training_config.output_dir) / "train_config.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True))
    return str(yaml_path)


class LlamaFactoryRunner:
    """Runs LlamaFactory training as a subprocess and streams its logs."""

    def __init__(self, llamafactory_dir: str):
        self.llamafactory_dir = Path(llamafactory_dir)

    def is_available(self) -> bool:
        return self.llamafactory_dir.exists()

    def run(self, yaml_path: str) -> tuple[int, list[str]]:
        """Launch training. Returns (return_code, captured_log_lines)."""
        cmd = [
            "python",
            str(self.llamafactory_dir / "src" / "train.py"),
            yaml_path,
        ]

        log_lines: list[str] = []
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(self.llamafactory_dir),
        )

        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            logger.info(f"[LlamaFactory] {line}")
            log_lines.append(line)

        process.wait()
        return process.returncode, log_lines
