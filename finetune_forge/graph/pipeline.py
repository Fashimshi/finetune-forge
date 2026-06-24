# finetune_forge/graph/pipeline.py

import logging
from langgraph.graph import StateGraph, END
from finetune_forge.schemas.state import PipelineState
from finetune_forge.agents.planner import run_planner
from finetune_forge.agents.data_processor import run_data_processor
from finetune_forge.agents.configurator import run_configurator
from finetune_forge.agents.executor import run_executor
from finetune_forge.agents.evaluator import run_evaluator
from finetune_forge.agents.publisher import run_publisher

logger = logging.getLogger(__name__)


def _should_abort(state: PipelineState) -> str:
    """Conditional edge: route to END if error, else continue."""
    if state.get("error"):
        logger.error(f"Pipeline aborting at step '{state.get('current_step')}': {state['error']}")
        return "abort"
    return "continue"


def build_pipeline() -> StateGraph:
    """
    Builds the FineTuneForge LangGraph DAG.

    Flow:
        planner → data_processor → configurator → executor → evaluator → publisher → END
        Any node that sets state['error'] routes to END immediately.
    """
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("planner", run_planner)
    graph.add_node("data_processor", run_data_processor)
    graph.add_node("configurator", run_configurator)
    graph.add_node("executor", run_executor)
    graph.add_node("evaluator", run_evaluator)
    graph.add_node("publisher", run_publisher)

    # Entry point
    graph.set_entry_point("planner")

    # Edges with error routing
    for src, dst in [
        ("planner", "data_processor"),
        ("data_processor", "configurator"),
        ("configurator", "executor"),
        ("executor", "evaluator"),
        ("evaluator", "publisher"),
    ]:
        graph.add_conditional_edges(
            src,
            _should_abort,
            {"continue": dst, "abort": END},
        )

    graph.add_edge("publisher", END)

    return graph.compile()


def run_pipeline(
    task_description: str,
    dataset_path: str,
    output_hub_repo: str,
) -> PipelineState:
    """
    Main entry point. Runs the full FineTuneForge pipeline.

    Args:
        task_description: Plain English description of the fine-tuning task.
        dataset_path: Path to local dataset file (.json/.jsonl/.csv) or HF dataset ID.
        output_hub_repo: HuggingFace Hub repo ID, e.g. "username/my-model".

    Returns:
        Final PipelineState with all results.
    """
    pipeline = build_pipeline()

    initial_state: PipelineState = {
        "task_description": task_description,
        "dataset_path": dataset_path,
        "output_hub_repo": output_hub_repo,
        "model_config": None,
        "available_vram_gb": None,
        "training_config": None,
        "llamafactory_yaml_path": None,
        "dataset_info": None,
        "training_complete": False,
        "checkpoint_path": None,
        "training_logs": None,
        "evaluation_result": None,
        "hub_url": None,
        "error": None,
        "current_step": "init",
    }

    final_state = pipeline.invoke(initial_state)
    return final_state
