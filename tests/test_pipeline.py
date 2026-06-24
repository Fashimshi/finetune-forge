# tests/test_pipeline.py
"""Integration test: full graph run with LLM + executor + publisher mocked."""

import finetune_forge.agents.planner as planner_mod
import finetune_forge.agents.configurator as cfg_mod
import finetune_forge.agents.evaluator as eval_mod
import finetune_forge.agents.executor as exec_mod
from finetune_forge.graph.pipeline import run_pipeline


def test_full_pipeline_happy_path(monkeypatch, tmp_path, sft_dataset_file):
    # Planner: fixed VRAM + model choice.
    monkeypatch.setattr(planner_mod, "get_available_vram_gb", lambda: 16.0)
    monkeypatch.setattr(
        planner_mod,
        "call_llm",
        lambda **kwargs: {
            "model_name": "microsoft/Phi-3.5-mini-instruct",
            "model_size_b": 3.8,
            "reasoning": "small + fits",
        },
    )

    # Configurator: fixed hyperparams, output into tmp_path.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cfg_mod,
        "call_llm",
        lambda **kwargs: {
            "num_epochs": 1,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "max_seq_length": 2048,
            "fp16": False,
            "bf16": True,
            "logging_steps": 10,
            "save_strategy": "epoch",
        },
    )

    # Executor: skip real training, just mark complete.
    def fake_executor(state):
        state["current_step"] = "executor"
        state["training_complete"] = True
        state["checkpoint_path"] = str(tmp_path / "output")
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        state["training_logs"] = "ok"
        return state

    monkeypatch.setattr(exec_mod, "run_executor", fake_executor)
    # The graph imported the symbol directly, so patch it there too.
    import finetune_forge.graph.pipeline as pipe_mod
    monkeypatch.setattr(pipe_mod, "run_executor", fake_executor)

    # Evaluator: deterministic judge.
    monkeypatch.setattr(
        eval_mod,
        "call_llm",
        lambda **kwargs: {"score": 9.0, "reasoning": "great"},
    )

    # Publisher: no real upload.
    def fake_publisher(state):
        state["current_step"] = "publisher"
        state["hub_url"] = "https://huggingface.co/tester/support-assistant"
        return state

    monkeypatch.setattr(pipe_mod, "run_publisher", fake_publisher)

    final = run_pipeline(
        task_description="Answer customer support questions.",
        dataset_path=sft_dataset_file,
        output_hub_repo="tester/support-assistant",
    )

    assert final["error"] is None
    assert final["model_config"].model_name == "microsoft/Phi-3.5-mini-instruct"
    assert final["training_complete"] is True
    assert final["evaluation_result"].judge_score == 0.9
    assert final["hub_url"].endswith("support-assistant")


def test_pipeline_aborts_on_planner_error(monkeypatch, sft_dataset_file):
    monkeypatch.setattr(planner_mod, "get_available_vram_gb", lambda: 0.0)

    def boom(**kwargs):
        raise ValueError("no json")

    monkeypatch.setattr(planner_mod, "call_llm", boom)

    final = run_pipeline(
        task_description="x",
        dataset_path=sft_dataset_file,
        output_hub_repo="",
    )
    # Planner error should short-circuit the graph; later steps never run.
    assert final["error"] is not None
    assert final["dataset_info"] is None
