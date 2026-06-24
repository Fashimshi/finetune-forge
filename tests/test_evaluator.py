# tests/test_evaluator.py

import json

import finetune_forge.agents.evaluator as eval_mod
from finetune_forge.agents.evaluator import run_evaluator, _load_sample_outputs
from finetune_forge.schemas.state import DatasetInfo


def _make_dataset_info(tmp_path) -> DatasetInfo:
    p = tmp_path / "processed_dataset.json"
    p.write_text(json.dumps([{"instruction": "q", "input": "", "output": "a"}]))
    return DatasetInfo(raw_path="x", format="sft", processed_path=str(p), num_examples=1)


def test_load_sample_outputs(tmp_path):
    di = _make_dataset_info(tmp_path)
    out = _load_sample_outputs(di, n=5)
    assert len(out) == 1


def test_load_sample_outputs_missing():
    assert _load_sample_outputs(None) == []


def test_run_evaluator_scores(monkeypatch, base_state, tmp_path):
    base_state["checkpoint_path"] = str(tmp_path)
    base_state["dataset_info"] = _make_dataset_info(tmp_path)

    monkeypatch.setattr(
        eval_mod,
        "call_llm",
        lambda **kwargs: {"score": 8.0, "reasoning": "good"},
    )

    out = run_evaluator(base_state)
    er = out["evaluation_result"]
    assert er.judge_score == 0.8
    assert er.passed is True


def test_run_evaluator_zero_score_is_not_passed(monkeypatch, base_state, tmp_path):
    base_state["checkpoint_path"] = str(tmp_path)
    base_state["dataset_info"] = _make_dataset_info(tmp_path)
    monkeypatch.setattr(eval_mod, "call_llm", lambda **kwargs: {"score": 0.0, "reasoning": "bad"})

    out = run_evaluator(base_state)
    er = out["evaluation_result"]
    # A real 0.0 score must be preserved (not coerced to a default) and fail the gate.
    assert er.judge_score == 0.0
    assert er.passed is False


def test_run_evaluator_no_checkpoint(base_state):
    base_state["checkpoint_path"] = None
    out = run_evaluator(base_state)
    assert "no checkpoint_path" in out["error"]


def test_run_evaluator_survives_judge_failure(monkeypatch, base_state, tmp_path):
    base_state["checkpoint_path"] = str(tmp_path)
    base_state["dataset_info"] = _make_dataset_info(tmp_path)

    def boom(**kwargs):
        raise RuntimeError("judge down")

    monkeypatch.setattr(eval_mod, "call_llm", boom)
    out = run_evaluator(base_state)
    # Evaluation failure must not block the pipeline.
    assert out["error"] is None
    assert out["evaluation_result"].passed is True
