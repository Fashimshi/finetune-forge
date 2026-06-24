# finetune_forge/utils/gpu.py

import subprocess
import logging

logger = logging.getLogger(__name__)


def get_available_vram_gb() -> float:
    """
    Returns total free VRAM in GB across all GPUs.
    Falls back to 0.0 if no GPU is detected (CPU-only mode).
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        free_mib_values = [int(v.strip()) for v in result.stdout.strip().split("\n") if v.strip()]
        total_free_mib = sum(free_mib_values)
        return round(total_free_mib / 1024, 2)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not detect GPU VRAM: {e}. Assuming CPU-only mode.")
        return 0.0


def get_gpu_count() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return len([line for line in result.stdout.strip().split("\n") if line.strip()])
    except Exception:
        return 0


MODEL_VRAM_REQUIREMENTS = {
    # (model_size_b, method) -> minimum VRAM in GB
    (3.8, "lora"):    6.0,
    (3.8, "qlora"):   4.0,
    (3.8, "full_ft"): 16.0,
    (7.0, "lora"):    12.0,
    (7.0, "qlora"):   6.0,
    (7.0, "full_ft"): 40.0,
    (8.0, "lora"):    14.0,
    (8.0, "qlora"):   8.0,
    (8.0, "full_ft"): 48.0,
    (13.0, "lora"):   20.0,
    (13.0, "qlora"):  12.0,
    (13.0, "full_ft"): 80.0,
    (70.0, "lora"):   48.0,
    (70.0, "qlora"):  24.0,
}


def _nearest_known_size(model_size_b: float) -> float:
    """Snap an arbitrary model size to the nearest size bucket we have data for."""
    known_sizes = sorted({size for (size, _method) in MODEL_VRAM_REQUIREMENTS})
    return min(known_sizes, key=lambda s: abs(s - model_size_b))


def get_feasible_method(model_size_b: float, available_vram_gb: float) -> str:
    """
    Returns the highest-quality feasible training method for the given model size and VRAM.
    Priority: full_ft > lora > qlora. Returns 'qlora' as last resort.

    The model size is snapped to the nearest known bucket so that arbitrary
    parameter counts (e.g. 3.82B) still resolve to a requirement.
    """
    size = _nearest_known_size(model_size_b)
    for method in ("full_ft", "lora", "qlora"):
        required = MODEL_VRAM_REQUIREMENTS.get((size, method))
        if required is not None and available_vram_gb >= required:
            return method
    return "qlora"  # always possible with enough quantization
