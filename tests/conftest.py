# tests/conftest.py

import json
from pathlib import Path

import pytest

from finetune_forge.schemas.state import PipelineState


@pytest.fixture
def base_state() -> PipelineState:
    """A fresh, fully-initialised PipelineState for node-level tests."""
    return {
        "task_description": "Answer customer support questions for a SaaS product.",
        "dataset_path": "",
        "output_hub_repo": "tester/support-assistant",
        "model_config": None,
        "available_vram_gb": None,
        "training_config": None,
        "llamafactory_yaml_path": None,
        "dataset_info": None,
        "hpo_trials": 0,
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


@pytest.fixture
def sft_dataset_file(tmp_path: Path) -> str:
    """Write a small alpaca-style SFT dataset and return its path."""
    rows = [
        {"instruction": "How do I reset my password?", "output": "Click 'Forgot password' on the login page."},
        {"instruction": "How do I cancel my subscription?", "output": "Go to Billing > Cancel plan."},
        {"instruction": "Where are my invoices?", "output": "They live under Billing > Invoices."},
    ]
    p = tmp_path / "support_qa.json"
    p.write_text(json.dumps(rows))
    return str(p)
