# tests/test_hpo.py

import math

import finetune_forge.agents.hpo as hpo_mod
from finetune_forge.agents.hpo import run_hpo, _objective_from_records, FAILED_TRIAL_LOSS
from finetune_forge.schemas.state import ModelConfig, TrainingConfig, DatasetInfo


def _prep_state(base_state, tmp_path, dataset_file, *, trials):
    base_state["model_config"] = ModelConfig(
        model_name="test/model", model_size_b=3.8, training_method="lora"
    )
    base_state["training_config"] = TrainingConfig(
        output_dir=str(tmp_path / "output"), learning_rate=2e-4
    )
    base_state["dataset_info"] = DatasetInfo(
        raw_path=dataset_file, format="sft", processed_path=dataset_file, num_examples=3
    )
    base_state["dataset_path"] = dataset_file
    base_state["hpo_trials"] = trials
    return base_state


def test_hpo_skips_when_disabled(base_state):
    base_state["hpo_trials"] = 0
    out = run_hpo(base_state)
    assert out["hpo_result"] is None
    assert out["current_step"] == "hpo"


def test_hpo_skips_without_config(base_state):
    base_state["hpo_trials"] = 5
    base_state["model_config"] = None
    out = run_hpo(base_state)
    assert out["hpo_result"] is None


def test_hpo_search_converges_and_applies(monkeypatch, base_state, tmp_path, sft_dataset_file):
    _prep_state(base_state, tmp_path, sft_dataset_file, trials=15)

    # Deterministic proxy objective: loss minimised at lr=1e-4 and batch_size=2.
    def fake_trial(*, params, **_kwargs):
        lr_term = (math.log10(params["learning_rate"]) - math.log10(1e-4)) ** 2
        batch_term = abs(params["per_device_train_batch_size"] - 2) * 0.5
        return lr_term + batch_term

    monkeypatch.setattr(hpo_mod, "_evaluate_trial", fake_trial)

    out = run_hpo(base_state)

    hpo = out["hpo_result"]
    assert hpo is not None
    assert hpo.num_trials == 15
    assert len(hpo.trials) == 15
    # Optuna should land near the optimum.
    assert hpo.best_params["per_device_train_batch_size"] == 2
    assert 3e-5 < hpo.best_params["learning_rate"] < 3e-4

    # Best params are written back into the live config + a fresh YAML is produced.
    assert out["training_config"].learning_rate == hpo.best_params["learning_rate"]
    assert out["training_config"].per_device_train_batch_size == 2
    assert out["model_config"].lora_rank == hpo.best_params["lora_rank"]
    assert out["model_config"].lora_alpha == hpo.best_params["lora_rank"] * 2
    assert out["llamafactory_yaml_path"] is not None


def test_hpo_handles_all_failed_trials(monkeypatch, base_state, tmp_path, sft_dataset_file):
    _prep_state(base_state, tmp_path, sft_dataset_file, trials=3)
    monkeypatch.setattr(hpo_mod, "_evaluate_trial", lambda **_kw: FAILED_TRIAL_LOSS)

    out = run_hpo(base_state)
    # No usable trial -> baseline config preserved, result records zero successes.
    assert out["hpo_result"].best_params == {}
    assert out["training_config"].learning_rate == 2e-4


def test_objective_from_records_prefers_eval_loss():
    records = [{"loss": 1.5}, {"loss": 1.2, "eval_loss": 1.3}, {"eval_loss": 1.0}]
    assert _objective_from_records(records) == 1.0


def test_objective_from_records_falls_back_to_loss():
    assert _objective_from_records([{"loss": 0.8}, {"loss": 0.9}]) == 0.8


def test_objective_from_records_penalises_no_signal():
    assert _objective_from_records([{"grad_norm": 0.1}]) == FAILED_TRIAL_LOSS


def test_evaluate_trial_unavailable_backend(monkeypatch, tmp_path, sft_dataset_file):
    # When LlamaFactory isn't present, a trial returns the failure penalty.
    class _Unavailable:
        def __init__(self, *a, **k):
            pass

        def is_available(self):
            return False

    monkeypatch.setattr(hpo_mod, "LlamaFactoryRunner", _Unavailable)
    loss = hpo_mod._evaluate_trial(
        params={"learning_rate": 2e-4, "per_device_train_batch_size": 2},
        model_config=ModelConfig(model_name="m", model_size_b=3.8),
        training_config=TrainingConfig(output_dir=str(tmp_path)),
        dataset_path=sft_dataset_file,
        llamafactory_dir="/nonexistent",
        trial_number=0,
    )
    assert loss == FAILED_TRIAL_LOSS
