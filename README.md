# FineTuneForge — AI Agent for Automated LLM Fine-Tuning

A multi-agent orchestration system that autonomously plans, configures, executes, and
evaluates LLM fine-tuning jobs across model sizes — from Phi-3.5 Mini (3.8B) to
LLaMA 3.1 (70B+). Built on **LlamaFactory** as the training backend, with **LangGraph**
powering the agent DAG.

```
plan → preprocess → configure → train → evaluate → publish
```

---

## What it does

- Accepts a plain-English task description + a dataset and returns a fine-tuned model
  pushed to the HuggingFace Hub.
- Automatically selects model, method (LoRA / QLoRA / full fine-tuning), and
  hyperparameters based on the task and available VRAM.
- Validates and reformats datasets into the correct training schema (SFT, DPO, reward).
- Tracks experiments via MLflow and evaluates with an LLM-as-judge scorer post-training.
- Exposes a clean CLI (`finetune run`) and a Python API (`run_pipeline`).

## Architecture

Six agents wired into a `langgraph.StateGraph`. Each node reads and writes a shared
`PipelineState` TypedDict; any node that sets `state["error"]` short-circuits the DAG to
`END`.

| Agent | Module | Responsibility |
|---|---|---|
| Planner | `agents/planner.py` | Task + VRAM → model & training-method selection |
| DataProcessor | `agents/data_processor.py` | Validate, detect format, convert to alpaca schema |
| Configurator | `agents/configurator.py` | Generate hyperparameters + LlamaFactory YAML |
| Executor | `agents/executor.py` | Run LlamaFactory training as a subprocess |
| Evaluator | `agents/evaluator.py` | LLM-as-judge scoring of outputs |
| Publisher | `agents/publisher.py` | Push model + card to HuggingFace Hub |

The LlamaFactory-specific YAML builder and subprocess runner live in
`backends/llamafactory.py`, keeping the agents backend-agnostic.

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
```

Python API:

```python
from finetune_forge.graph.pipeline import run_pipeline

state = run_pipeline(
    task_description="Answer customer support questions",
    dataset_path="./data/support_qa.jsonl",
    output_hub_repo="your-username/support-assistant-v1",
)
print(state["hub_url"])
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
| `LLAMAFACTORY_DIR` | Path to the cloned LlamaFactory repo |

Example LlamaFactory configs the Configurator emits live in `configs/examples/`.

## Status & roadmap

Phase 1 (this repo) is the end-to-end agent pipeline. Planned Phase 2 work:

- MLflow logging of loss curves / VRAM in the executor
- Hyperparameter-optimization agent (Optuna pilots)
- Resumable pipeline via LangGraph persistence
- Dataset curator agent (search the HF Datasets Hub)
- Gradio/Streamlit dashboard

### Known limitations

- The Executor requires a local LlamaFactory checkout and a registered
  `dataset_info.json` entry mapping `custom_dataset` to the processed file — LlamaFactory's
  dataset-registration step is not yet automated.
- The judge currently scores training-set outputs as a proxy; it does not yet run
  inference against the freshly trained checkpoint.

## Tech stack

LangGraph · LlamaFactory · HuggingFace (PEFT / TRL / Hub / Datasets) · bitsandbytes ·
MLflow · Pydantic · Typer · Rich · pytest.
