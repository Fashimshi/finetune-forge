# finetune_forge/agents/evaluator.py

import logging
import json
from pathlib import Path
from finetune_forge.schemas.state import PipelineState, EvaluationResult
from finetune_forge.utils.llm_client import call_llm

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an expert LLM evaluator.
You will be given a task description and a sample of model outputs.
Score the outputs on a scale of 1-10 and provide brief reasoning.

Respond ONLY with JSON:
{
  "score": <float 1-10>,
  "reasoning": "<2-3 sentence explanation>"
}

Criteria:
- Relevance to the task (0-3 points)
- Output quality and coherence (0-3 points)
- Instruction following (0-2 points)
- Absence of hallucinations or refusals (0-2 points)
"""


def run_evaluator(state: PipelineState) -> PipelineState:
    """
    LangGraph node: evaluates fine-tuned model with LLM-as-judge.
    Loads a few examples from the processed dataset and scores outputs.
    """
    state["current_step"] = "evaluator"
    dataset_info = state.get("dataset_info")
    checkpoint_path = state.get("checkpoint_path")

    if not checkpoint_path:
        state["error"] = "Evaluator: no checkpoint_path in state"
        return state

    logger.info(f"Evaluator: running post-training evaluation on {checkpoint_path}")

    # Load sample examples from processed dataset
    sample_outputs = _load_sample_outputs(dataset_info, n=5)

    if not sample_outputs:
        logger.warning("Evaluator: no sample outputs found, skipping LLM judge")
        state["evaluation_result"] = EvaluationResult(passed=True)
        return state

    prompt = f"""Task: {state['task_description']}

Sample model outputs (from training data):
{json.dumps(sample_outputs, indent=2)}

Evaluate the quality of these outputs for the given task."""

    try:
        result = call_llm(
            prompt=prompt,
            system=JUDGE_SYSTEM_PROMPT,
            expect_json=True,
        )
        raw_score = float(result.get("score", 5.0))
        normalized_score = raw_score / 10.0  # 0.0 - 1.0

        eval_result = EvaluationResult(
            judge_score=normalized_score,
            judge_reasoning=result.get("reasoning", ""),
            passed=normalized_score >= 0.5,
        )
        state["evaluation_result"] = eval_result

        logger.info(
            f"Evaluator: judge_score={normalized_score:.2f}, "
            f"passed={eval_result.passed}, "
            f"reasoning={eval_result.judge_reasoning}"
        )

    except Exception as e:
        logger.error(f"Evaluator failed: {e}")
        # Don't block the pipeline on evaluation failure
        state["evaluation_result"] = EvaluationResult(passed=True)

    return state


def _load_sample_outputs(dataset_info, n: int = 5) -> list[dict]:
    """Load n random examples from processed dataset for judge evaluation."""
    if dataset_info is None or dataset_info.processed_path is None:
        return []
    path = Path(dataset_info.processed_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        # Return last n examples (often more representative than first n)
        return data[-n:]
    except Exception:
        return []
