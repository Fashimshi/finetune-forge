# FineTuneForge — AI Agent for Automated LLM Fine-Tuning

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-48%20passing-brightgreen.svg)](tests/)

A multi-agent orchestration system that autonomously plans, configures, optimizes,
executes, and evaluates LLM fine-tuning jobs across model sizes — from Phi-3.5 Mini
(3.8B) to LLaMA 3.1 (70B+). Built on **LlamaFactory** as the training backend, with
**LangGraph** powering the agent DAG.

```
plan → preprocess → configure → [tune] → train → evaluate → publish
```

The optional `[tune]` step is an Optuna hyperparameter search; the whole run is
checkpointed so it can resume from the last completed step after an interruption.

---

## What it does

- Accepts a plain-English task description + a dataset and returns a fine-tuned model
  pushed to the HuggingFace Hub.
- Automatically selects model, method (LoRA / QLoRA / full fine-tuning), and
  hyperparameters based on the task and available VRAM.
- Validates and reformats datasets into the correct training schema (SFT, DPO, reward).
- Optionally runs an **Optuna hyperparameter search** (learning rate / batch size /
  LoRA rank) over short proxy trials before committing to the full run.
- Logs **loss curves, hyperparameters, and VRAM usage to MLflow** straight from the
  training logs, and evaluates with an LLM-as-judge scorer post-training.
- **Resumes interrupted runs** from the last completed step via LangGraph SQLite
  checkpointing — no work is repeated after a crash.
- Exposes a clean CLI (`finetune run`) and a Python API (`run_pipeline`).

## Architecture

Seven agents wired into a `langgraph.StateGraph`. Each node reads and writes a shared
`PipelineState` TypedDict; any node that sets `state["error"]` short-circuits the DAG to
`END`.

| Agent | Module | Responsibility |
|---|---|---|
| Planner | `agents/planner.py` | Task + VRAM → model & training-method selection |
| DataProcessor | `agents/data_processor.py` | Validate, detect format, convert to alpaca schema |
| Configurator | `agents/configurator.py` | Generate hyperparameters + LlamaFactory YAML |
| HPO | `agents/hpo.py` | Optional Optuna search over LR / batch / LoRA rank |
| Executor | `agents/executor.py` | Run LlamaFactory training; log metrics to MLflow |
| Evaluator | `agents/evaluator.py` | LLM-as-judge scoring of outputs |
| Publisher | `agents/publisher.py` | Push model + card to HuggingFace Hub |

The LlamaFactory-specific YAML builder and subprocess runner live in
`backends/llamafactory.py`, and MLflow log-parsing/tracking lives in
`utils/tracking.py`, keeping the agents backend- and tracker-agnostic.

The HPO node is a no-op unless `hpo_trials > 0`. When enabled it runs each Optuna
trial as a short proxy training run, reads the eval loss back from the logs, and writes
the winning hyperparameters into the state before the full training run.

## Install

```bash
# Clone this repo, then the training backend.
git clone https://github.com/hiyouga/LlamaFactory
pip install -e LlamaFactory

# Install FineTuneForge (+ dev tools for the test suite).
pip install -e ".[dev]"

cp .env.example .env   # fill in ANTHROPIC_API_KEY, HF_TOKEN, LLAMAFACTORY_DIR
```

Requires Python 3.11+.

## Usage

```bash
# Check environment (GPU, tokens, LlamaFactory path).
finetune info

# Run the full pipeline.
finetune run \
  --task "Fine-tune a model to answer customer support questions for a SaaS product" \
  --dataset ./data/support_qa.jsonl \
  --hub-repo your-username/support-assistant-v1

# Tune hyperparameters first (10 Optuna proxy trials), and checkpoint the run so it
# can resume after an interruption.
finetune run \
  --task "Answer customer support questions" \
  --dataset ./data/support_qa.jsonl \
  --hub-repo your-username/support-assistant-v1 \
  --hpo-trials 10 \
  --thread-id support-v1

# Resume that run from its last completed step.
finetune run \
  --task "Answer customer support questions" \
  --dataset ./data/support_qa.jsonl \
  --thread-id support-v1 \
  --resume
```

Python API:

```python
from finetune_forge.graph.pipeline import run_pipeline

state = run_pipeline(
    task_description="Answer customer support questions",
    dataset_path="./data/support_qa.jsonl",
    output_hub_repo="your-username/support-assistant-v1",
    hpo_trials=10,           # optional Optuna search; 0 disables it
    thread_id="support-v1",  # checkpoint the run so resume=True can pick it back up
)
print(state["hub_url"])
print(state["training_metrics"])  # {'final_loss': ..., 'min_eval_loss': ...}
print(state["mlflow_run_id"])
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite mocks the LLM calls, the training subprocess, and the Hub upload, so it runs
without a GPU, API keys, or network access.

## Configuration

Environment variables (see `.env.example`):

| Var | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API for planner / configurator / judge |
| `HF_TOKEN` | HuggingFace Hub upload |
| `OPENAI_API_KEY` | Optional, for a GPT-4o judge |
| `MLFLOW_TRACKING_URI` | MLflow experiment tracking (default `./mlruns`) |
| `FINETUNE_CHECKPOINT_DB` | SQLite path for resumable runs (default `./.finetune_forge/checkpoints.sqlite`) |
| `LLAMAFACTORY_DIR` | Path to the cloned LlamaFactory repo |

Example LlamaFactory configs the Configurator emits live in `configs/examples/`.

## Status & roadmap

**Phase 1** — the end-to-end agent pipeline (plan → preprocess → configure → train →
evaluate → publish).

**Phase 2 (this release)** — shipped:

- ✅ MLflow logging of loss curves / VRAM, parsed from the training logs in the executor
- ✅ Hyperparameter-optimization agent (Optuna proxy trials)
- ✅ Resumable pipeline via LangGraph SQLite checkpointing

**Phase 3** — planned:

- Dataset curator agent (search the HF Datasets Hub)
- Gradio/Streamlit dashboard
- Judge inference against the freshly trained checkpoint (vs. the current proxy)

### Known limitations

- The Executor requires a local LlamaFactory checkout (`LLAMAFACTORY_DIR`). The
  Configurator automatically writes the companion `dataset_info.json` registering the
  processed dataset, so no manual registration is needed.
- The HPO agent's proxy trials also require the LlamaFactory backend; with no backend
  available it logs a warning and leaves the baseline config untouched rather than failing.
- The judge currently scores training-set outputs as a proxy; it does not yet run
  inference against the freshly trained checkpoint.

## Tech stack

LangGraph (+ SQLite checkpointer) · LlamaFactory · HuggingFace (PEFT / TRL / Hub /
Datasets) · bitsandbytes · Optuna · MLflow · Pydantic · Typer · Rich · pytest.
