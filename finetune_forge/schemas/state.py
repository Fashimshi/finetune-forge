# finetune_forge/schemas/state.py

from __future__ import annotations
from typing import TypedDict, Optional, Literal
from pydantic import BaseModel, Field


TrainingMethod = Literal["lora", "qlora", "full_ft", "dpo", "reward_modeling"]
DatasetFormat = Literal["sft", "dpo", "reward", "pretrain"]


class ModelConfig(BaseModel):
    model_name: str = Field(..., description="HuggingFace model ID, e.g. 'meta-llama/Llama-3.1-8B'")
    model_size_b: float = Field(..., description="Parameter count in billions")
    quantization: Optional[Literal["4bit", "8bit", "none"]] = "none"
    training_method: TrainingMethod = "lora"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    num_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048
    fp16: bool = False
    bf16: bool = True
    output_dir: str = "./output"
    logging_steps: int = 10
    save_strategy: str = "epoch"
    dataloader_num_workers: int = 4


class DatasetInfo(BaseModel):
    raw_path: str
    format: DatasetFormat
    processed_path: Optional[str] = None
    num_examples: Optional[int] = None
    quality_score: Optional[float] = None
    issues: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    perplexity: Optional[float] = None
    rouge_l: Optional[float] = None
    judge_score: Optional[float] = None
    judge_reasoning: Optional[str] = None
    passed: bool = False


class PipelineState(TypedDict):
    # Inputs
    task_description: str
    dataset_path: str
    output_hub_repo: str               # e.g. "username/my-finetuned-model"

    # Computed by Planner
    model_config: Optional[ModelConfig]
    available_vram_gb: Optional[float]

    # Computed by Configurator
    training_config: Optional[TrainingConfig]
    llamafactory_yaml_path: Optional[str]

    # Computed by DataProcessor
    dataset_info: Optional[DatasetInfo]

    # Computed by Executor
    training_complete: bool
    checkpoint_path: Optional[str]
    training_logs: Optional[str]

    # Computed by Evaluator
    evaluation_result: Optional[EvaluationResult]

    # Computed by Publisher
    hub_url: Optional[str]

    # Error handling
    error: Optional[str]
    current_step: str
