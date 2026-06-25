# finetune_forge/agents/hpo.py
"""Hyperparameter-optimization agent.

Runs a short Optuna search over the most impactful knobs (learning rate, batch
size, LoRA rank) using *proxy* trials -- a handful of training steps each -- then
writes the best configuration back into the pipeline state before the full run.

The agent is opt-in: it only runs when ``state['hpo_trials'] > 0``. It also
degrades gracefully -- if Optuna isn't installed or the training backend is
unavailable, it leaves the baseline config untouched and lets the pipeline
continue rather than aborting.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from finetune_forge.schemas.state import PipelineState, HPOResult, ModelConfig, TrainingConfig
from finetune_forge.backends.llamafactory import LlamaFactoryRunner, build_llamafactory_yaml
from finetune_forge.utils.tracking import parse_training_metrics

logger = logging.getLogger(__name__)

# Each proxy trial runs only this many steps -- enough signal, cheap to run.
PROXY_MAX_STEPS = 30
# Penalty objective returned when a trial fails to produce a usable loss.
FAILED_TRIAL_LOSS = 1e9
# Fixed sampler seed so a given search is reproducible.
_SEED = 42


def run_hpo(state: PipelineState) -> PipelineState:
    """LangGraph node: optionally tune hyperparameters with Optuna proxy trials."""
    state["current_step"] = "hpo"
    trials = int(state.get("hpo_trials") or 0)

    if trials <= 0:
        logger.info("HPO: hpo_trials=0, skipping hyperparameter search.")
        return state

    model_config = state.get("model_config")
    training_config = state.get("training_config")
    if model_config is None or training_config is None:
        logger.warning("HPO: missing model_config/training_config, skipping search.")
        return state

    try:
        import optuna
    except ImportError:
        logger.warning("HPO: optuna not installed, skipping search (pip install optuna).")
        return state

    dataset_path = _resolve_dataset_path(state)
    llamafactory_dir = os.environ.get("LLAMAFACTORY_DIR", "./LlamaFactory")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=_SEED),
    )

    def objective(trial) -> float:
        params = _suggest_params(trial, model_config)
        return _evaluate_trial(
            params=params,
            model_config=model_config,
            training_config=training_config,
            dataset_path=dataset_path,
            llamafactory_dir=llamafactory_dir,
            trial_number=trial.number,
        )

    logger.info(f"HPO: starting Optuna search ({trials} trials, {PROXY_MAX_STEPS} steps each)")
    try:
        study.optimize(objective, n_trials=trials)
    except Exception as e:
        logger.error(f"HPO: optimization failed: {e}; keeping baseline config.")
        return state

    completed = [t for t in study.trials if t.value is not None and t.value < FAILED_TRIAL_LOSS]
    if not completed:
        logger.warning("HPO: no successful trials; keeping baseline config.")
        state["hpo_result"] = HPOResult(num_trials=trials, trials=[])
        return state

    best = study.best_trial
    logger.info(f"HPO: best params {best.params} (eval loss {best.value:.4f})")

    # Write tuned values back into the configs and rebuild the main YAML.
    _apply_best_params(state, best.params)

    state["hpo_result"] = HPOResult(
        best_params=best.params,
        best_value=float(best.value),
        num_trials=trials,
        direction="minimize",
        trials=[{"params": t.params, "value": t.value} for t in study.trials],
    )
    return state


def _suggest_params(trial, model_config: ModelConfig) -> dict:
    """Define the search space sampled for each trial."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [1, 2, 4]
        ),
    }
    # LoRA rank only matters for adapter-based methods.
    if model_config.training_method in ("lora", "qlora"):
        params["lora_rank"] = trial.suggest_categorical("lora_rank", [8, 16, 32, 64])
    return params


def _evaluate_trial(
    *,
    params: dict,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    dataset_path: str,
    llamafactory_dir: str,
    trial_number: int,
) -> float:
    """Run one short proxy training run and return its eval loss (lower is better)."""
    runner = LlamaFactoryRunner(llamafactory_dir)
    if not runner.is_available():
        logger.warning("HPO: LlamaFactory unavailable; trial cannot run.")
        return FAILED_TRIAL_LOSS

    trial_model = model_config
    if "lora_rank" in params:
        rank = int(params["lora_rank"])
        trial_model = model_config.model_copy(update={"lora_rank": rank, "lora_alpha": rank * 2})

    trial_train = training_config.model_copy(update={
        "learning_rate": float(params["learning_rate"]),
        "per_device_train_batch_size": int(params["per_device_train_batch_size"]),
        "output_dir": str(Path(training_config.output_dir) / "hpo" / f"trial_{trial_number}"),
    })

    yaml_path = build_llamafactory_yaml(
        model_config=trial_model,
        training_config=trial_train,
        dataset_path=dataset_path,
        extra_overrides={
            "max_steps": PROXY_MAX_STEPS,
            "save_steps": PROXY_MAX_STEPS,
            "eval_steps": max(PROXY_MAX_STEPS // 2, 1),
        },
        yaml_name="trial_config.yaml",
    )

    return_code, log_lines = runner.run(yaml_path)
    if return_code != 0:
        logger.warning(f"HPO: trial {trial_number} failed (exit {return_code}).")
        return FAILED_TRIAL_LOSS

    records = parse_training_metrics(log_lines, output_dir=trial_train.output_dir)
    return _objective_from_records(records)


def _objective_from_records(records: list[dict]) -> float:
    """Best eval loss if available, else best train loss, else a failure penalty."""
    eval_losses = [r["eval_loss"] for r in records if isinstance(r.get("eval_loss"), (int, float))]
    if eval_losses:
        return float(min(eval_losses))
    train_losses = [r["loss"] for r in records if isinstance(r.get("loss"), (int, float))]
    if train_losses:
        return float(min(train_losses))
    return FAILED_TRIAL_LOSS


def _apply_best_params(state: PipelineState, best_params: dict) -> None:
    """Merge the winning trial's params into model/training config and rebuild the YAML."""
    model_config = state["model_config"]
    training_config = state["training_config"]

    if "lora_rank" in best_params:
        rank = int(best_params["lora_rank"])
        model_config = model_config.model_copy(update={"lora_rank": rank, "lora_alpha": rank * 2})
        state["model_config"] = model_config

    train_updates: dict = {}
    if "learning_rate" in best_params:
        train_updates["learning_rate"] = float(best_params["learning_rate"])
    if "per_device_train_batch_size" in best_params:
        train_updates["per_device_train_batch_size"] = int(best_params["per_device_train_batch_size"])
    if train_updates:
        training_config = training_config.model_copy(update=train_updates)
        state["training_config"] = training_config

    yaml_path = build_llamafactory_yaml(
        model_config=model_config,
        training_config=training_config,
        dataset_path=_resolve_dataset_path(state),
    )
    state["llamafactory_yaml_path"] = yaml_path


def _resolve_dataset_path(state: PipelineState) -> str:
    """Prefer the processed dataset; fall back to the raw input path."""
    dataset_info = state.get("dataset_info")
    if dataset_info and dataset_info.processed_path:
        return dataset_info.processed_path
    return state["dataset_path"]
