# finetune_forge/agents/executor.py

import logging
import os
from pathlib import Path

from finetune_forge.schemas.state import PipelineState
from finetune_forge.backends.llamafactory import LlamaFactoryRunner

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

    except Exception as e:
        logger.error(f"Executor: unexpected error: {e}")
        state["error"] = f"Executor error: {str(e)}"

    return state
