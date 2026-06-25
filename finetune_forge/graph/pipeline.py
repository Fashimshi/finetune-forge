# finetune_forge/graph/pipeline.py

import logging
import os
from pathlib import Path
from typing import Optional

from langgraph.graph import StateGraph, END

from finetune_forge.schemas.state import PipelineState
from finetune_forge.agents.planner import run_planner
from finetune_forge.agents.data_processor import run_data_processor
from finetune_forge.agents.configurator import run_configurator
from finetune_forge.agents.hpo import run_hpo
from finetune_forge.agents.executor import run_executor
from finetune_forge.agents.evaluator import run_evaluator
from finetune_forge.agents.publisher import run_publisher

logger = logging.getLogger(__name__)

# Default on-disk location for the LangGraph checkpointer (used for resume).
DEFAULT_CHECKPOINT_DB = "./.finetune_forge/checkpoints.sqlite"


def _should_abort(state: PipelineState) -> str:
    """Conditional edge: route to END if error, else continue."""
    if state.get("error"):
        logger.error(f"Pipeline aborting at step '{state.get('current_step')}': {state['error']}")
        return "abort"
    return "continue"


def build_pipeline(checkpointer=None):
    """
    Builds the FineTuneForge LangGraph DAG.

    Flow:
        planner → data_processor → configurator → hpo → executor → evaluator → publisher → END
        Any node that sets state['error'] routes to END immediately.

    When a ``checkpointer`` is supplied, the compiled graph persists state after
    every node so an interrupted run can be resumed (see ``run_pipeline(resume=True)``).
    """
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("planner", run_planner)
    graph.add_node("data_processor", run_data_processor)
    graph.add_node("configurator", run_configurator)
    graph.add_node("hpo", run_hpo)
    graph.add_node("executor", run_executor)
    graph.add_node("evaluator", run_evaluator)
    graph.add_node("publisher", run_publisher)

    # Entry point
    graph.set_entry_point("planner")

    # Edges with error routing
    for src, dst in [
        ("planner", "data_processor"),
        ("data_processor", "configurator"),
        ("configurator", "hpo"),
        ("hpo", "executor"),
        ("executor", "evaluator"),
        ("evaluator", "publisher"),
    ]:
        graph.add_conditional_edges(
            src,
            _should_abort,
            {"continue": dst, "abort": END},
        )

    graph.add_edge("publisher", END)

    return graph.compile(checkpointer=checkpointer)


def _initial_state(
    task_description: str,
    dataset_path: str,
    output_hub_repo: str,
    hpo_trials: int,
) -> PipelineState:
    return {
        "task_description": task_description,
        "dataset_path": dataset_path,
        "output_hub_repo": output_hub_repo,
        "model_config": None,
        "available_vram_gb": None,
        "training_config": None,
        "llamafactory_yaml_path": None,
        "dataset_info": None,
        "hpo_trials": hpo_trials,
        "hpo_result": None,
        "training_complete": False,
        "checkpoint_path": None,
        "training_logs": None,
        "training_metrics": None,
        "mlflow_run_id": None,
        "evaluation_result": None,
        "hub_url": None,
        "error": None,
        "current_step": "init",
    }


def run_pipeline(
    task_description: str,
    dataset_path: str,
    output_hub_repo: str,
    hpo_trials: int = 0,
    thread_id: Optional[str] = None,
    resume: bool = False,
    checkpoint_db: Optional[str] = None,
) -> PipelineState:
    """
    Main entry point. Runs the full FineTuneForge pipeline.

    Args:
        task_description: Plain English description of the fine-tuning task.
        dataset_path: Path to local dataset file (.json/.jsonl/.csv) or HF dataset ID.
        output_hub_repo: HuggingFace Hub repo ID, e.g. "username/my-model".
        hpo_trials: If > 0, run an Optuna hyperparameter search before training.
        thread_id: Stable id used to checkpoint this run so it can be resumed.
        resume: Resume an interrupted run identified by ``thread_id`` from its
            last saved checkpoint instead of starting over.
        checkpoint_db: Override the SQLite checkpoint path (defaults to env
            FINETUNE_CHECKPOINT_DB or ``./.finetune_forge/checkpoints.sqlite``).

    Returns:
        Final PipelineState with all results.
    """
    use_checkpointer = bool(thread_id) or resume

    if not use_checkpointer:
        pipeline = build_pipeline()
        return pipeline.invoke(
            _initial_state(task_description, dataset_path, output_hub_repo, hpo_trials)
        )

    # Resumable run: persist state to SQLite under a stable thread id.
    from langgraph.checkpoint.sqlite import SqliteSaver

    db = checkpoint_db or os.environ.get("FINETUNE_CHECKPOINT_DB", DEFAULT_CHECKPOINT_DB)
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    tid = thread_id or "default"
    config = {"configurable": {"thread_id": tid}}

    with SqliteSaver.from_conn_string(db) as checkpointer:
        pipeline = build_pipeline(checkpointer=checkpointer)
        if resume:
            logger.info(f"Resuming pipeline thread '{tid}' from checkpoint {db}")
            # Passing None continues the existing thread from its last checkpoint.
            return pipeline.invoke(None, config=config)
        logger.info(f"Starting checkpointed pipeline thread '{tid}' at {db}")
        return pipeline.invoke(
            _initial_state(task_description, dataset_path, output_hub_repo, hpo_trials),
            config=config,
        )
