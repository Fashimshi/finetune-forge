# finetune_forge/utils/tracking.py
"""MLflow experiment tracking: turn raw training logs into metric series and log them.

Kept out of the executor so the agent stays backend-agnostic, and written to
degrade gracefully: if MLflow isn't installed or anything goes wrong, tracking
becomes a logged no-op rather than failing the training run.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# HF Trainer prints python-dict-ish metric blobs to stdout, e.g.
#   {'loss': 1.2345, 'learning_rate': 0.0002, 'epoch': 0.5}
#   {'eval_loss': 1.10, 'eval_runtime': 3.2, 'epoch': 1.0}
_METRIC_DICT_RE = re.compile(r"\{[^{}]*'(?:loss|eval_loss)'[^{}]*\}")

# Numeric per-step keys we surface as MLflow metric curves.
_TRACKED_KEYS = ("loss", "eval_loss", "learning_rate", "grad_norm", "epoch")


def parse_training_metrics(log_lines: list[str], output_dir: Optional[str] = None) -> list[dict]:
    """Extract per-step metric records from a finished training run.

    Prefers LlamaFactory's structured ``trainer_log.jsonl`` (one JSON object per
    logged step) when present in ``output_dir``; otherwise falls back to scraping
    the HF-Trainer dict blobs out of the captured stdout lines.
    """
    if output_dir:
        jsonl = Path(output_dir) / "trainer_log.jsonl"
        if jsonl.exists():
            records: list[dict] = []
            for line in jsonl.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if records:
                return records

    records = []
    for line in log_lines or []:
        for match in _METRIC_DICT_RE.findall(line):
            try:
                parsed = ast.literal_eval(match)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, dict) and any(k in parsed for k in ("loss", "eval_loss")):
                records.append(parsed)
    return records


def summarize_metrics(records: list[dict]) -> dict:
    """Reduce per-step records to a flat summary (final/min train & eval loss)."""
    losses = [r["loss"] for r in records if isinstance(r.get("loss"), (int, float))]
    eval_losses = [r["eval_loss"] for r in records if isinstance(r.get("eval_loss"), (int, float))]

    summary: dict = {"num_log_steps": len(records)}
    if losses:
        summary["final_loss"] = float(losses[-1])
        summary["min_loss"] = float(min(losses))
    if eval_losses:
        summary["final_eval_loss"] = float(eval_losses[-1])
        summary["min_eval_loss"] = float(min(eval_losses))
    return summary


def log_to_mlflow(
    *,
    run_name: str,
    params: dict,
    metric_records: list[dict],
    summary: dict,
    tracking_uri: Optional[str] = None,
) -> Optional[str]:
    """Log params, per-step loss/lr curves, and summary metrics to MLflow.

    Returns the MLflow run id, or ``None`` if tracking was skipped (MLflow
    missing) or failed — experiment tracking must never break a training run.
    """
    try:
        import mlflow
    except ImportError:
        logger.info("MLflow not installed; skipping experiment tracking.")
        return None

    try:
        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
        # MLflow 3.x rejects the local file store unless explicitly opted in; keep
        # the zero-config ``./mlruns`` default working without a database backend.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("finetune-forge")

        with mlflow.start_run(run_name=run_name) as run:
            loggable_params = {k: v for k, v in (params or {}).items() if v is not None}
            if loggable_params:
                mlflow.log_params(loggable_params)

            for fallback_step, rec in enumerate(metric_records or []):
                step_num = int(rec.get("current_steps") or rec.get("step") or fallback_step)
                for key in _TRACKED_KEYS:
                    val = rec.get(key)
                    if isinstance(val, (int, float)):
                        mlflow.log_metric(key, float(val), step=step_num)

            numeric_summary = {k: float(v) for k, v in (summary or {}).items() if isinstance(v, (int, float))}
            if numeric_summary:
                mlflow.log_metrics(numeric_summary)

            logger.info(f"MLflow: logged run {run.info.run_id} to {uri}")
            return run.info.run_id
    except Exception as e:  # pragma: no cover - defensive; tracking never blocks training
        logger.warning(f"MLflow logging failed: {e}")
        return None
