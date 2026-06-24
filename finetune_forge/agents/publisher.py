# finetune_forge/agents/publisher.py

import logging
from pathlib import Path
from huggingface_hub import HfApi
from finetune_forge.schemas.state import PipelineState

logger = logging.getLogger(__name__)


def run_publisher(state: PipelineState) -> PipelineState:
    """
    LangGraph node: uploads model checkpoint to HuggingFace Hub.
    Requires HF_TOKEN env variable.
    """
    state["current_step"] = "publisher"
    checkpoint_path = state.get("checkpoint_path")
    hub_repo = state.get("output_hub_repo")

    if not checkpoint_path or not Path(checkpoint_path).exists():
        state["error"] = f"Publisher: checkpoint not found at {checkpoint_path}"
        return state

    if state.get("model_config") is None:
        state["error"] = "Publisher: no model_config in state"
        return state

    if not hub_repo:
        logger.warning("Publisher: no output_hub_repo specified, skipping upload")
        return state

    logger.info(f"Publisher: uploading {checkpoint_path} to {hub_repo}")

    try:
        api = HfApi()

        # Create repo if it doesn't exist
        api.create_repo(repo_id=hub_repo, exist_ok=True, private=True)

        # Upload folder
        api.upload_folder(
            folder_path=checkpoint_path,
            repo_id=hub_repo,
            commit_message=f"FineTuneForge: fine-tuned {state['model_config'].model_name}",
        )

        # Generate model card
        _push_model_card(api, state, hub_repo)

        hub_url = f"https://huggingface.co/{hub_repo}"
        state["hub_url"] = hub_url
        logger.info(f"Publisher: model uploaded to {hub_url}")

    except Exception as e:
        logger.error(f"Publisher failed: {e}")
        state["error"] = f"Publisher error: {str(e)}"

    return state


def _push_model_card(api: HfApi, state: PipelineState, hub_repo: str) -> None:
    """Generates and pushes a minimal model card README."""
    model_config = state["model_config"]
    eval_result = state.get("evaluation_result")
    judge_score = (
        f"{eval_result.judge_score:.2f}"
        if eval_result and eval_result.judge_score is not None
        else "N/A"
    )

    card_content = f"""---
base_model: {model_config.model_name}
tags:
  - finetune-forge
  - lora
  - {model_config.training_method}
---

# {hub_repo.split('/')[-1]}

Fine-tuned from [{model_config.model_name}](https://huggingface.co/{model_config.model_name})
using [FineTuneForge](https://github.com/your-username/finetune-forge).

## Training Details

| Parameter | Value |
|---|---|
| Base Model | {model_config.model_name} |
| Method | {model_config.training_method.upper()} |
| Quantization | {model_config.quantization} |
| LoRA Rank | {model_config.lora_rank} |
| LLM Judge Score | {judge_score} / 1.0 |

## Task

{state['task_description']}
"""
    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=hub_repo,
        commit_message="Add model card",
    )
