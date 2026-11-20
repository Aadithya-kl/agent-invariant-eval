"""Command line interface: ``agent-eval scan`` and ``agent-eval benchmark``."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Annotated, Any

try:
    import typer
except ImportError as exc:
    raise ImportError(
        "The CLI requires the 'cli' extra: pip install 'agent-invariant-eval[cli]'"
    ) from exc

from agent_eval.benchmark.frontier import FrontierRunner, ModelConfig, load_dataset
from agent_eval.decorators import SimClock
from agent_eval.invariants import default_registry, find_pci, find_pii

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Deterministic, fail-closed evaluation for multi-agent workflows.",
)


def _load_object(reference: str) -> Any:
    """Import ``package.module:attribute``. The current directory is importable."""
    module_name, _, attribute = reference.partition(":")
    if not module_name or not attribute:
        raise typer.BadParameter(f"expected 'module:attribute', got {reference!r}")
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    try:
        return getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise typer.BadParameter(f"cannot load {reference!r}: {exc}") from exc


def _load_models(reference: str) -> list[ModelConfig]:
    """Model configurations from a JSON file or a ``module:attribute`` list."""
    path = Path(reference)
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"{reference}: not valid JSON: {exc}") from exc
    elif ":" in reference:
        raw = _load_object(reference)
    else:
        raise typer.BadParameter(
            f"{reference!r} is neither an existing JSON file nor a 'module:attribute' reference"
        )
    if not isinstance(raw, (list, tuple)) or not raw:
        raise typer.BadParameter(f"{reference}: expected a non empty list of model configurations")
    try:
        return [ModelConfig.model_validate(item) for item in raw]
    except ValueError as exc:
        raise typer.BadParameter(f"{reference}: invalid model configuration: {exc}") from exc


@app.command()
def scan(
    text: Annotated[str | None, typer.Argument(help="Text to scan. Omit to use --file.")] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f", help="File to scan.")] = None,
) -> None:
    """Scan text for PII and PCI data. Findings are printed masked. Exit 1 on any finding."""
    if (text is None) == (file is None):
        raise typer.BadParameter("give either TEXT or --file, not both")
    content = text if text is not None else file.read_text(encoding="utf-8")  # type: ignore[union-attr]
    findings = [*find_pii(content), *find_pci(content)]
    if not findings:
        typer.echo("clean: no PII or PCI findings")
        return
    for finding in sorted(findings, key=lambda f: f.start):
        typer.echo(f"{finding.kind:<20} offset {finding.start:<6} {finding.masked}")
    raise typer.Exit(code=1)


@app.command()
def benchmark(
    dataset: Annotated[Path, typer.Argument(help="JSONL evaluation dataset.")],
    agent: Annotated[
        str, typer.Option(help="Agent runner as 'module:callable' taking (case, model, session).")
    ],
    models: Annotated[
        str,
        typer.Option(
            help="Model configurations: a JSON file, or 'module:attribute' naming a list."
        ),
    ],
    registry: Annotated[
        str | None,
        typer.Option(help="Registry factory as 'module:callable'. Defaults to the built in set."),
    ] = None,
    min_pass_rate: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.9,
    sim_clock: Annotated[
        bool, typer.Option(help="Use a simulated clock (for agents that call clock.advance).")
    ] = False,
    max_concurrency: Annotated[int, typer.Option(min=1, help="Cases to run at once.")] = 1,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the full report as JSON.")
    ] = None,
) -> None:
    """Replay a dataset across model configurations and print the Pareto table."""
    configs = _load_models(models)
    invariants = _load_object(registry)() if registry else default_registry()
    runner = FrontierRunner(
        _load_object(agent),
        invariants=invariants,
        min_pass_rate=min_pass_rate,
        clock_factory=SimClock if sim_clock else None,
        max_concurrency=max_concurrency,
    )
    report = runner.run(load_dataset(dataset), configs)
    typer.echo(report.to_markdown())
    typer.echo(f"\nrecommended: {report.recommended}")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(f"report written to {output}")


if __name__ == "__main__":
    app()
