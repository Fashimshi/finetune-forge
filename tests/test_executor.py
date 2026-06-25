# tests/test_executor.py

import finetune_forge.agents.executor as exec_mod
from finetune_forge.agents.executor import run_executor
from finetune_forge.schemas.state import ModelConfig, TrainingConfig


class _FakeRunner:
    """Stand-in for LlamaFactoryRunner that skips real training."""

    def __init__(self, *_args, **_kwargs):
        pass

    def is_available(self) -> bool:
        return True

    def run(self, _yaml_path):
        log_lines = [
            "{'loss': 2.0, 'learning_rate': 0.0002, 'epoch': 0.5}",
            "{'loss': 1.2, 'learning_rate': 0.0001, 'epoch': 1.0}",
            "{'eval_loss': 1.1, 'epoch': 1.0}",
        ]
        return 0, log_lines


def _prep_state(base_state, tmp_path):
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    yaml_path = tmp_path / "train_config.yaml"
    yaml_path.write_text("stage: sft\n")

    base_state["model_config"] = ModelConfig(model_name="test/model", model_size_b=3.8)
    base_state["training_config"] = TrainingConfig(output_dir=str(out_dir))
    base_state["llamafactory_yaml_path"] = str(yaml_path)
    return base_state


def test_executor_parses_metrics_and_tracks(monkeypatch, base_state, tmp_path):
    _prep_state(base_state, tmp_path)
    monkeypatch.setattr(exec_mod, "LlamaFactoryRunner", _FakeRunner)
    monkeypatch.setattr(exec_mod, "get_available_vram_gb", lambda: 24.0)

    captured = {}

    def fake_log(**kwargs):
        captured.update(kwargs)
        return "run-abc"

    monkeypatch.setattr(exec_mod, "log_to_mlflow", fake_log)

    out = run_executor(base_state)

    assert out["error"] is None
    assert out["training_complete"] is True
    # Loss curve was parsed into a summary.
    assert out["training_metrics"]["final_loss"] == 1.2
    assert out["training_metrics"]["min_eval_loss"] == 1.1
    # MLflow received params + the per-step records + a VRAM signal.
    assert out["mlflow_run_id"] == "run-abc"
    assert captured["params"]["model_name"] == "test/model"
    assert captured["summary"]["vram_used_gb"] == 0.0  # before == after in this test


def test_executor_training_failure_skips_tracking(monkeypatch, base_state, tmp_path):
    _prep_state(base_state, tmp_path)

    class _FailRunner(_FakeRunner):
        def run(self, _yaml_path):
            return 1, ["boom", "traceback"]

    monkeypatch.setattr(exec_mod, "LlamaFactoryRunner", _FailRunner)

    def should_not_run(**kwargs):
        raise AssertionError("tracking must not run on training failure")

    monkeypatch.setattr(exec_mod, "log_to_mlflow", should_not_run)

    out = run_executor(base_state)
    assert "exit code 1" in out["error"]
    assert out["training_complete"] is False
