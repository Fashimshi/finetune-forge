# finetune_forge/agents/configurator.py

import logging

from finetune_forge.schemas.state import PipelineState, TrainingConfig
from finetune_forge.utils.llm_client import call_llm
from finetune_forge.backends.llamafactory import build_llamafactory_yaml

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert in LLM fine-tuning configuration.
Given model and task details, return ONLY a JSON object with optimal TrainingConfig values.

Schema:
{
  "num_epochs": int,
  "per_device_train_batch_size": int,
  "gradient_accumulation_steps": int,
  "learning_rate": float,
  "lr_scheduler_type": str,
  "warmup_ratio": float,
  "max_seq_length": int,
  "fp16": bool,
  "bf16": bool,
  "logging_steps": int,
  "save_strategy": str
}

Rules:
- For small models (< 5B): batch_size=4, lr=2e-4
- For medium models (5-15B): batch_size=2, gradient_accumulation=8, lr=1e-4
- For large models (> 15B): batch_size=1, gradient_accumulation=16, lr=5e-5
- bf16=true if VRAM >= 16GB, else fp16=true
- max_seq_length: 2048 for chat, 4096 for long-context tasks
- Prefer cosine scheduler; use linear for very small datasets (< 500 examples)
"""


def run_configurator(state: PipelineState) -> PipelineState:
    """
    LangGraph node: generates LlamaFactory YAML config.
    """
    state["current_step"] = "configurator"
    model_config = state["model_config"]
    dataset_info = state["dataset_info"]

    if model_config is None:
        state["error"] = "Configurator: no model_config found in state"
        return state

    logger.info(f"Configurator: generating training config for {model_config.model_name}")

    prompt = f"""Model: {model_config.model_name} ({model_config.model_size_b}B)
Training method: {model_config.training_method}
Quantization: {model_config.quantization}
Available VRAM: {state.get('available_vram_gb', 0)}GB
Dataset size: {dataset_info.num_examples if dataset_info else 'unknown'} examples
Dataset format: {dataset_info.format if dataset_info else 'sft'}
Task: {state['task_description']}

Generate optimal training hyperparameters."""

    try:
        result = call_llm(prompt=prompt, system=SYSTEM_PROMPT, expect_json=True)
        training_config = TrainingConfig(**result)
        state["training_config"] = training_config

        # Build LlamaFactory YAML via the backend.
        dataset_path = (
            dataset_info.processed_path
            if dataset_info and dataset_info.processed_path
            else state["dataset_path"]
        )
        yaml_path = build_llamafactory_yaml(
            model_config=model_config,
            training_config=training_config,
            dataset_path=dataset_path,
        )
        state["llamafactory_yaml_path"] = yaml_path
        logger.info(f"Configurator: wrote LlamaFactory config to {yaml_path}")

    except Exception as e:
        logger.error(f"Configurator failed: {e}")
        state["error"] = f"Configurator error: {str(e)}"

    return state
