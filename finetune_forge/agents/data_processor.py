# finetune_forge/agents/data_processor.py

import json
import logging
from pathlib import Path
from datasets import load_dataset, Dataset
from finetune_forge.schemas.state import PipelineState, DatasetInfo
from finetune_forge.schemas.dataset import (
    REQUIRED_SFT_KEYS,
    REQUIRED_DPO_KEYS,
)

logger = logging.getLogger(__name__)


def run_data_processor(state: PipelineState) -> PipelineState:
    """
    LangGraph node: validates dataset, detects format, converts if needed.
    Expects state['dataset_path'] to be a local path or HF dataset ID.
    """
    state["current_step"] = "data_processor"
    dataset_path = state["dataset_path"]
    logger.info(f"DataProcessor: loading dataset from {dataset_path}")

    issues: list[str] = []
    processed_path: str | None = None

    try:
        # Load dataset
        dataset = _load_dataset(dataset_path)
        num_examples = len(dataset)

        # Detect format
        detected_format = _detect_format(dataset)
        logger.info(f"DataProcessor: detected format={detected_format}, n={num_examples}")

        # Validate
        issues = _validate(dataset, detected_format)
        if issues:
            logger.warning(f"DataProcessor: found {len(issues)} issues: {issues}")

        # Convert to LlamaFactory SFT format (alpaca-style)
        converted = _convert_to_alpaca(dataset, detected_format)

        # Save processed dataset
        out_path = Path(dataset_path).parent / "processed_dataset.json"
        out_path.write_text(json.dumps(converted, indent=2, ensure_ascii=False))
        processed_path = str(out_path)

        quality_score = _compute_quality_score(converted)
        logger.info(f"DataProcessor: quality_score={quality_score:.2f}, saved to {processed_path}")

        state["dataset_info"] = DatasetInfo(
            raw_path=dataset_path,
            format=detected_format,
            processed_path=processed_path,
            num_examples=num_examples,
            quality_score=quality_score,
            issues=issues,
        )

    except Exception as e:
        logger.error(f"DataProcessor failed: {e}")
        state["error"] = f"DataProcessor error: {str(e)}"

    return state


def _load_dataset(path: str) -> Dataset:
    """Load from local file or HuggingFace Hub."""
    p = Path(path)
    if p.exists() and p.suffix in (".json", ".jsonl"):
        return Dataset.from_json(str(p))
    if p.exists() and p.suffix == ".csv":
        return Dataset.from_csv(str(p))
    # Attempt HF Hub
    return load_dataset(path, split="train")


def _detect_format(dataset: Dataset) -> str:
    cols = set(dataset.column_names)
    if REQUIRED_DPO_KEYS.issubset(cols):
        return "dpo"
    if REQUIRED_SFT_KEYS.issubset(cols):
        return "sft"
    if {"text"}.issubset(cols):
        return "pretrain"
    # Try to infer from common alternatives
    if {"question", "answer"}.issubset(cols):
        return "sft"
    if {"input", "output"}.issubset(cols):
        return "sft"
    return "sft"  # default


def _validate(dataset: Dataset, fmt: str) -> list[str]:
    issues = []
    cols = set(dataset.column_names)

    if fmt == "sft":
        for key in REQUIRED_SFT_KEYS:
            if key not in cols:
                # Check for aliases
                if key == "instruction" and "question" not in cols and "input" not in cols:
                    issues.append(f"Missing column: '{key}' (or 'question'/'input' alias)")
                elif key == "output" and "answer" not in cols:
                    issues.append(f"Missing column: '{key}' (or 'answer' alias)")

    if fmt == "dpo":
        for key in REQUIRED_DPO_KEYS:
            if key not in cols:
                issues.append(f"Missing column: '{key}'")

    # Check for empty rows
    sample = dataset.select(range(min(100, len(dataset))))
    first_col = dataset.column_names[0]
    empty_count = sum(1 for row in sample if not row[first_col])
    if empty_count > 0:
        issues.append(f"Found {empty_count} empty rows in first 100 samples")

    return issues


def _convert_to_alpaca(dataset: Dataset, fmt: str) -> list[dict]:
    """Convert dataset to LlamaFactory alpaca format for SFT."""
    result = []

    if fmt == "sft":
        for row in dataset:
            instruction = row.get("instruction") or row.get("question") or row.get("input") or ""
            output = row.get("output") or row.get("answer") or ""
            context = row.get("context") or row.get("system") or ""
            result.append({
                "instruction": instruction,
                "input": context,
                "output": output,
            })

    elif fmt == "dpo":
        for row in dataset:
            result.append({
                "instruction": row.get("prompt", ""),
                "chosen": row.get("chosen", ""),
                "rejected": row.get("rejected", ""),
            })

    elif fmt == "pretrain":
        for row in dataset:
            result.append({"text": row.get("text", "")})

    else:
        # Passthrough
        result = [dict(row) for row in dataset]

    return result


def _compute_quality_score(data: list[dict]) -> float:
    """
    Heuristic quality score 0.0-1.0 based on:
    - Average output length (longer = more signal)
    - Fraction of non-empty outputs
    - No duplicate instructions
    """
    if not data:
        return 0.0

    outputs = [str(row.get("output", row.get("chosen", row.get("text", "")))) for row in data]
    instructions = [str(row.get("instruction", "")) for row in data]

    non_empty_frac = sum(1 for o in outputs if o.strip()) / len(outputs)
    avg_len = sum(len(o.split()) for o in outputs) / max(len(outputs), 1)
    len_score = min(avg_len / 100.0, 1.0)  # 100 words = perfect length score

    unique_frac = len(set(instructions)) / max(len(instructions), 1)

    score = (non_empty_frac * 0.4) + (len_score * 0.3) + (unique_frac * 0.3)
    return round(score, 3)
