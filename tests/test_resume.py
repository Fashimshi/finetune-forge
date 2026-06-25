# tests/test_resume.py
"""Resumable-pipeline tests: a checkpointed run survives a mid-pipeline crash."""

import pytest

import finetune_forge.graph.pipeline as pipe_mod
from finetune_forge.graph.pipeline import run_pipeline
from finetune_forge.schemas.state import ModelConfig, TrainingConfig, DatasetInfo, EvaluationResult


def _install_fakes(monkeypatch, counters, fail_executor_once):
    """Replace every node with a light fake; executor crashes on its first call."""

    def planner(state):
        counters["planner"] += 1
        state["current_step"] = "planner"
        state["model_config"] = ModelConfig(model_name="test/model", model_size_b=3.8)
        return state

    def data_processor(state):
        counters["data_processor"] += 1
        state["current_step"] = "data_processor"
        state["dataset_info"] = DatasetInfo(raw_path="x", format="sft", num_examples=3)
        return state

    def configurator(state):
        counters["configurator"] += 1
        state["current_step"] = "configurator"
        state["training_config"] = TrainingConfig()
        state["llamafactory_yaml_path"] = "/tmp/train.yaml"
        return state

    def executor(state):
        counters["executor"] += 1
        if fail_executor_once["pending"]:
            fail_executor_once["pending"] = False
            raise RuntimeError("simulated crash mid-training")
        state["current_step"] = "executor"
        state["training_complete"] = True
        state["checkpoint_path"] = "/tmp/output"
        return state

    def evaluator(state):
        counters["evaluator"] += 1
        state["current_step"] = "evaluator"
        state["evaluation_result"] = EvaluationResult(judge_score=0.9, passed=True)
        return state

    def publisher(state):
        counters["publisher"] += 1
        state["current_step"] = "publisher"
        state["hub_url"] = "https://huggingface.co/tester/model"
        return state

    monkeypatch.setattr(pipe_mod, "run_planner", planner)
    monkeypatch.setattr(pipe_mod, "run_data_processor", data_processor)
    monkeypatch.setattr(pipe_mod, "run_configurator", configurator)
    monkeypatch.setattr(pipe_mod, "run_executor", executor)
    monkeypatch.setattr(pipe_mod, "run_evaluator", evaluator)
    monkeypatch.setattr(pipe_mod, "run_publisher", publisher)


def test_resume_continues_after_crash(monkeypatch, tmp_path, sft_dataset_file):
    counters = {k: 0 for k in
                ["planner", "data_processor", "configurator", "executor", "evaluator", "publisher"]}
    fail_once = {"pending": True}
    _install_fakes(monkeypatch, counters, fail_once)

    db = str(tmp_path / "checkpoints.sqlite")

    # First run crashes inside the executor node.
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_pipeline(
            task_description="t",
            dataset_path=sft_dataset_file,
            output_hub_repo="tester/model",
            thread_id="job-1",
            checkpoint_db=db,
        )

    assert counters["planner"] == 1
    assert counters["configurator"] == 1
    assert counters["executor"] == 1  # attempted once, crashed

    # Resume from the saved checkpoint; the executor re-runs and succeeds.
    final = run_pipeline(
        task_description="t",
        dataset_path=sft_dataset_file,
        output_hub_repo="tester/model",
        thread_id="job-1",
        resume=True,
        checkpoint_db=db,
    )

    assert final["error"] is None
    assert final["training_complete"] is True
    assert final["hub_url"].endswith("tester/model")

    # Completed-before-crash nodes are NOT re-run on resume.
    assert counters["planner"] == 1
    assert counters["data_processor"] == 1
    assert counters["configurator"] == 1
    # Executor ran a second time (the retry); downstream nodes ran once.
    assert counters["executor"] == 2
    assert counters["evaluator"] == 1
    assert counters["publisher"] == 1


def test_checkpointed_run_completes_without_resume(monkeypatch, tmp_path, sft_dataset_file):
    counters = {k: 0 for k in
                ["planner", "data_processor", "configurator", "executor", "evaluator", "publisher"]}
    _install_fakes(monkeypatch, counters, {"pending": False})

    final = run_pipeline(
        task_description="t",
        dataset_path=sft_dataset_file,
        output_hub_repo="tester/model",
        thread_id="job-2",
        checkpoint_db=str(tmp_path / "cp.sqlite"),
    )
    assert final["error"] is None
    assert final["hub_url"].endswith("tester/model")
    assert counters["executor"] == 1
