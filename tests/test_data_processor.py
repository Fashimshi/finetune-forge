# tests/test_data_processor.py

import json
from pathlib import Path

from finetune_forge.agents.data_processor import (
    run_data_processor,
    _compute_quality_score,
)


def test_run_data_processor_sft(base_state, sft_dataset_file):
    base_state["dataset_path"] = sft_dataset_file
    out = run_data_processor(base_state)

    assert out["error"] is None
    di = out["dataset_info"]
    assert di is not None
    assert di.format == "sft"
    assert di.num_examples == 3
    assert di.quality_score is not None and di.quality_score > 0

    # Processed file exists and is alpaca-shaped.
    processed = json.loads(Path(di.processed_path).read_text())
    assert set(processed[0].keys()) == {"instruction", "input", "output"}


def test_quality_score_empty():
    assert _compute_quality_score([]) == 0.0


def test_quality_score_rewards_unique_nonempty():
    good = [{"instruction": f"q{i}", "output": "word " * 100} for i in range(10)]
    dup = [{"instruction": "same", "output": ""} for _ in range(10)]
    assert _compute_quality_score(good) > _compute_quality_score(dup)


def test_missing_file_records_error(base_state):
    base_state["dataset_path"] = "/nonexistent/does_not_exist.json"
    out = run_data_processor(base_state)
    # Falls through to HF Hub load attempt, which fails -> error recorded.
    assert out["error"] is not None
