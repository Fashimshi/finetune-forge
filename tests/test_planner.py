# tests/test_planner.py

import finetune_forge.agents.planner as planner_mod
from finetune_forge.agents.planner import run_planner, _get_target_modules
from finetune_forge.utils.gpu import get_feasible_method


def test_get_target_modules_by_family():
    assert "gate_proj" in _get_target_modules("meta-llama/Llama-3.1-8B-Instruct")
    assert _get_target_modules("microsoft/Phi-3.5-mini-instruct") == ["q_proj", "v_proj", "k_proj", "o_proj"]
    assert _get_target_modules("some/unknown-model") == ["q_proj", "v_proj"]


def test_get_feasible_method_prefers_highest_quality():
    # 8B model with 64GB -> full fine-tune feasible.
    assert get_feasible_method(8.0, 64.0) == "full_ft"
    # 8B model with 14GB -> lora.
    assert get_feasible_method(8.0, 14.0) == "lora"
    # 8B model with 8GB -> qlora.
    assert get_feasible_method(8.0, 8.0) == "qlora"
    # Tiny VRAM always falls back to qlora.
    assert get_feasible_method(70.0, 1.0) == "qlora"


def test_unknown_size_snaps_to_nearest_bucket():
    # 14B isn't an exact key; should snap to 13.0 and resolve a real method.
    assert get_feasible_method(14.0, 64.0) == "lora"


def test_run_planner_sets_model_config(monkeypatch, base_state):
    monkeypatch.setattr(planner_mod, "get_available_vram_gb", lambda: 16.0)
    monkeypatch.setattr(
        planner_mod,
        "call_llm",
        lambda **kwargs: {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "model_size_b": 8.0,
            "reasoning": "fits",
        },
    )

    out = run_planner(base_state)

    assert out["error"] is None
    assert out["available_vram_gb"] == 16.0
    mc = out["model_config"]
    assert mc is not None
    assert mc.model_name == "meta-llama/Llama-3.1-8B-Instruct"
    assert mc.training_method == "lora"  # 8B @ 16GB
    assert "gate_proj" in mc.target_modules


def test_run_planner_records_error_on_llm_failure(monkeypatch, base_state):
    monkeypatch.setattr(planner_mod, "get_available_vram_gb", lambda: 16.0)

    def boom(**kwargs):
        raise ValueError("bad json")

    monkeypatch.setattr(planner_mod, "call_llm", boom)

    out = run_planner(base_state)
    assert out["model_config"] is None
    assert "Planner error" in out["error"]
