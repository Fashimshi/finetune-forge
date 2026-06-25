# tests/test_tracking.py

import json
import sys

from finetune_forge.utils.tracking import (
    parse_training_metrics,
    summarize_metrics,
    log_to_mlflow,
)

# A couple of HF-Trainer stdout blobs as they appear in captured logs.
_STDOUT_LINES = [
    "Some banner line that is not a metric",
    "{'loss': 2.5, 'learning_rate': 0.0002, 'epoch': 0.5}",
    "{'loss': 1.8, 'learning_rate': 0.0001, 'epoch': 1.0}",
    "{'eval_loss': 1.6, 'eval_runtime': 3.1, 'epoch': 1.0}",
]


def test_parse_metrics_from_stdout():
    records = parse_training_metrics(_STDOUT_LINES)
    assert len(records) == 3
    assert records[0]["loss"] == 2.5
    assert records[-1]["eval_loss"] == 1.6


def test_parse_metrics_prefers_trainer_log_jsonl(tmp_path):
    # The structured file should win over stdout scraping.
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    rows = [
        {"current_steps": 10, "loss": 0.9, "learning_rate": 2e-4},
        {"current_steps": 20, "loss": 0.7, "eval_loss": 0.65},
    ]
    (out_dir / "trainer_log.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    records = parse_training_metrics(_STDOUT_LINES, output_dir=str(out_dir))
    assert records == rows  # came from the jsonl, not the stdout blobs


def test_parse_metrics_empty():
    assert parse_training_metrics([]) == []
    assert parse_training_metrics(["nothing here"]) == []


def test_summarize_metrics():
    records = [
        {"loss": 2.5},
        {"loss": 1.8},
        {"eval_loss": 1.6},
        {"eval_loss": 1.4},
    ]
    summary = summarize_metrics(records)
    assert summary["num_log_steps"] == 4
    assert summary["final_loss"] == 1.8
    assert summary["min_loss"] == 1.8
    assert summary["final_eval_loss"] == 1.4
    assert summary["min_eval_loss"] == 1.4


def test_summarize_metrics_empty():
    assert summarize_metrics([]) == {"num_log_steps": 0}


def test_log_to_mlflow_real_run(tmp_path):
    # mlflow is installed in the test env: log to a temp file store and get a run id.
    records = [{"current_steps": 10, "loss": 1.0, "learning_rate": 2e-4}]
    run_id = log_to_mlflow(
        run_name="unit-test",
        params={"model_name": "test/model", "learning_rate": 2e-4, "skip_me": None},
        metric_records=records,
        summary={"final_loss": 1.0, "num_log_steps": 1},
        tracking_uri=f"file:{tmp_path / 'mlruns'}",
    )
    assert run_id is not None and isinstance(run_id, str)


def test_log_to_mlflow_missing_mlflow_is_noop(monkeypatch):
    # Simulate mlflow not being installed -> graceful None, no exception.
    monkeypatch.setitem(sys.modules, "mlflow", None)
    run_id = log_to_mlflow(
        run_name="x", params={}, metric_records=[], summary={},
    )
    assert run_id is None
