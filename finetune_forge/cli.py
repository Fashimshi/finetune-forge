# finetune_forge/cli.py

import typer
from rich.console import Console
from rich.table import Table

from finetune_forge.graph.pipeline import run_pipeline
from finetune_forge.utils.logging import setup_logging

app = typer.Typer(name="finetune", help="FineTuneForge — AI Agent for LLM Fine-Tuning")
console = Console()

setup_logging()


@app.command()
def run(
    task: str = typer.Option(..., "--task", "-t", help="Task description in plain English"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to dataset file or HF dataset ID"),
    hub_repo: str = typer.Option("", "--hub-repo", "-r", help="HuggingFace Hub repo ID (user/repo)"),
    hpo_trials: int = typer.Option(0, "--hpo-trials", help="Run N Optuna hyperparameter-search trials before training"),
    thread_id: str = typer.Option("", "--thread-id", help="Checkpoint id so the run can be resumed if interrupted"),
    resume: bool = typer.Option(False, "--resume", help="Resume an interrupted run identified by --thread-id"),
):
    """
    Run the full FineTuneForge pipeline: plan → preprocess → configure → hpo → train → evaluate → push.
    """
    if resume:
        console.print(f"[bold green]FineTuneForge[/bold green] Resuming thread '{thread_id or 'default'}'...\n")
    else:
        console.print("[bold green]FineTuneForge[/bold green] Starting pipeline...\n")

    final_state = run_pipeline(
        task_description=task,
        dataset_path=dataset,
        output_hub_repo=hub_repo,
        hpo_trials=hpo_trials,
        thread_id=thread_id or None,
        resume=resume,
    )

    if final_state.get("error"):
        console.print(f"[bold red]Pipeline failed:[/bold red] {final_state['error']}")
        raise typer.Exit(code=1)

    # Print summary table
    table = Table(title="Pipeline Summary", show_header=True)
    table.add_column("Step", style="cyan")
    table.add_column("Result", style="green")

    mc = final_state.get("model_config")
    di = final_state.get("dataset_info")
    er = final_state.get("evaluation_result")
    hpo = final_state.get("hpo_result")
    tm = final_state.get("training_metrics")

    if mc:
        table.add_row("Model", f"{mc.model_name} ({mc.model_size_b}B)")
        table.add_row("Method", mc.training_method.upper())
        table.add_row("Quantization", mc.quantization)

    if di:
        table.add_row("Dataset", f"{di.num_examples} examples, quality={di.quality_score:.2f}")

    if hpo and hpo.best_params:
        table.add_row("HPO Best", f"{hpo.best_params} (loss={hpo.best_value:.4f})")

    if tm and tm.get("final_loss") is not None:
        table.add_row("Final Loss", f"{tm['final_loss']:.4f}")

    if final_state.get("mlflow_run_id"):
        table.add_row("MLflow Run", final_state["mlflow_run_id"])

    if er:
        table.add_row("Judge Score", f"{er.judge_score:.2f}/1.0" if er.judge_score else "N/A")
        table.add_row("Evaluation", "PASSED" if er.passed else "FAILED")

    if final_state.get("hub_url"):
        table.add_row("Hub URL", final_state["hub_url"])

    console.print(table)


@app.command()
def info():
    """Print environment info (GPU, LlamaFactory path)."""
    from finetune_forge.utils.gpu import get_available_vram_gb, get_gpu_count
    import os

    console.print(f"GPU count:       {get_gpu_count()}")
    console.print(f"Free VRAM:       {get_available_vram_gb()}GB")
    console.print(f"LlamaFactory:    {os.environ.get('LLAMAFACTORY_DIR', 'not set')}")
    console.print(f"HF_TOKEN:        {'set' if os.environ.get('HF_TOKEN') else 'not set'}")
    console.print(f"ANTHROPIC_KEY:   {'set' if os.environ.get('ANTHROPIC_API_KEY') else 'not set'}")


if __name__ == "__main__":
    app()
