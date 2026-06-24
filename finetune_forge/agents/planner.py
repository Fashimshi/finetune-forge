# finetune_forge/agents/planner.py

import logging
from finetune_forge.schemas.state import PipelineState, ModelConfig
from finetune_forge.utils.gpu import get_available_vram_gb, get_feasible_method
from finetune_forge.utils.llm_client import call_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert ML engineer specializing in LLM fine-tuning.
Given a task description and available GPU VRAM, you select the most appropriate
open-source model and training configuration.

Respond ONLY with valid JSON matching this schema:
{
  "model_name": "<huggingface model id>",
  "model_size_b": <float, parameter count in billions>,
  "reasoning": "<brief explanation of the choice>"
}

Model options to consider (pick the largest one feasible for the VRAM):
- microsoft/Phi-3.5-mini-instruct (3.8B) — best for < 8GB VRAM
- Qwen/Qwen3-7B (7B) — strong multilingual, great instruction following
- meta-llama/Llama-3.1-8B-Instruct (8B) — gold standard for SFT
- Qwen/Qwen3-14B (14B) — requires 20GB+ with LoRA
- meta-llama/Llama-3.1-70B-Instruct (70B) — only with 48GB+ VRAM for QLoRA

Rules:
- Prefer instruction-tuned base models for SFT tasks
- Prefer base (non-instruct) models for continued pre-training
- Always pick the largest model that fits the VRAM budget
- For VRAM < 8GB, always pick Phi-3.5-mini
"""


def run_planner(state: PipelineState) -> PipelineState:
    """
    LangGraph node: selects model + training method based on task and VRAM.
    """
    logger.info("Planner: detecting VRAM and selecting model...")

    vram_gb = get_available_vram_gb()
    state["available_vram_gb"] = vram_gb
    state["current_step"] = "planner"

    logger.info(f"Planner: detected {vram_gb}GB free VRAM")

    prompt = f"""Task description: {state['task_description']}

Available GPU VRAM: {vram_gb}GB

Select the best model for this fine-tuning task."""

    try:
        result = call_llm(prompt=prompt, system=SYSTEM_PROMPT, expect_json=True)

        model_name: str = result["model_name"]
        model_size_b: float = float(result["model_size_b"])

        training_method = get_feasible_method(model_size_b, vram_gb)
        quantization = "4bit" if training_method == "qlora" else ("8bit" if training_method == "lora" and vram_gb < 16 else "none")

        # Default LoRA target modules per model family
        target_modules = _get_target_modules(model_name)

        model_config = ModelConfig(
            model_name=model_name,
            model_size_b=model_size_b,
            quantization=quantization,
            training_method=training_method,
            target_modules=target_modules,
        )

        state["model_config"] = model_config
        logger.info(
            f"Planner: selected {model_name} ({model_size_b}B) "
            f"with method={training_method}, quant={quantization}"
        )

    except Exception as e:
        logger.error(f"Planner failed: {e}")
        state["error"] = f"Planner error: {str(e)}"

    return state


def _get_target_modules(model_name: str) -> list[str]:
    """Returns LoRA target modules for known model families."""
    name_lower = model_name.lower()
    if "llama" in name_lower:
        return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if "qwen" in name_lower:
        return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if "phi" in name_lower:
        return ["q_proj", "v_proj", "k_proj", "o_proj"]
    if "mistral" in name_lower or "mixtral" in name_lower:
        return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # Fallback
    return ["q_proj", "v_proj"]
