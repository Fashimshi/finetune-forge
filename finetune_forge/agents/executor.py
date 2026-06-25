# finetune_forge/agents/executor.py

import logging
import os
from pathlib import Path

from finetune_forge.schemas.state import PipelineState
from finetune_forge.backends.llamafactory import LlamaFactoryRunner
from finetune_forge.utils.gpu import get_available_vram_gb
from finetune_forge.utils.tracking import (
    parse_training_metrics,
    summarize_metrics,
    log_to_mlflow,
)

logger = logging.getLogger(__name__)


def run_executor(state: PipelineState) -> PipelineState:
    """
    LangGraph node: executes LlamaFactory training via subprocess.
    Reads LLAMAFACTORY_DIR from environment.
    """
    state["current_step"] = "executor"
    yaml_path = state.get("llamafactory_yaml_path")

    if not yaml_path or not Path(yaml_path).exists():
        state["error"] = f"Executor: YAML config not found at {yaml_path}"
        return state

    llamafactory_dir = os.environ.get("LLAMAFACTORY_DIR", "./LlamaFactory")
    runner = LlamaFactoryRunner(llamafactory_dir)
    if not runner.is_available():
        state["error"] = (
            f"Executor: LlamaFactory not found at {llamafactory_dir}. "
            "Clone it: git clone https://github.com/hiyouga/LlamaFactory"
        )
        return state

    logger.info(f"Executor: starting training with config {yaml_path}")

    vram_before = get_available_vram_gb()

    try:
        return_code, log_lines = runner.run(yaml_path)

        if return_code != 0:
            state["error"] = f"Executor: training failed with exit code {return_code}"
            state["training_logs"] = "\n".join(log_lines[-50:])  # Last 50 lines
            return state

        training_config = state["training_config"]
        checkpoint_path = str(Path(training_config.output_dir))
        state["training_complete"] = True
        state["checkpoint_path"] = checkpoint_path
        state["training_logs"] = "\n".join(log_lines[-100:])
        logger.info(f"Executor: training complete. Checkpoint at {checkpoint_path}")

        # Parse loss/eval-loss curves and ship params + metrics to MLflow.
        _track_run(state, log_lines, checkpoint_path, vram_before)

    except Exception as e:
        logger.error(f"Executor: unexpected error: {e}")
        state["error"] = f"Executor error: {str(e)}"

    return state


def _track_run(
    state: PipelineState,
    log_lines: list[str],
    checkpoint_path: str,
    vram_before: float,
) -> None:
    """Parse the loss curve from training logs and record it to MLflow.

    Any failure here is swallowed (and the summary is still saved to state) so
    experiment tracking can never turn a successful training run into a failure.
    """
    records = parse_training_metrics(log_lines, output_dir=checkpoint_path)
    summary = summarize_metrics(records)
    state["training_metrics"] = summary

    model_config = state.get("model_config")
    training_config = state["training_config"]
    vram_after = get_available_vram_gb()

    params = {
        "model_name": getattr(model_config, "model_name", None),
        "model_size_b": getattr(model_config, "model_size_b", None),
        "training_method": getattr(model_config, "training_method", None),
        "quantization": getattr(model_config, "quantization", None),
        "num_epochs": training_config.num_epochs,
        "learning_rate": training_config.learning_rate,
        "per_device_train_batch_size": training_config.per_device_train_batch_size,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
        "lr_scheduler_type": training_config.lr_scheduler_type,
        "max_seq_length": training_config.max_seq_length,
    }
    # VRAM signal: free memory before/after training and the implied peak usage.
    summary_with_vram = dict(summary)
    summary_with_vram["vram_free_gb_before"] = vram_before
    summary_with_vram["vram_free_gb_after"] = vram_after
    summary_with_vram["vram_used_gb"] = round(max(vram_before - vram_after, 0.0), 2)

    run_name = f"{getattr(model_config, 'model_name', 'model')}-{state.get('current_step', 'train')}"
    state["mlflow_run_id"] = log_to_mlflow(
        run_name=run_name,
        params=params,
        metric_records=records,
        summary=summary_with_vram,
    )
