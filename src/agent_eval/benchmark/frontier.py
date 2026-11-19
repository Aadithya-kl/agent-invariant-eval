"""Cost to performance frontier.

Replays one JSONL dataset across several model configurations, scores every
run with the deterministic invariants, and reports pass rate, total inference
cost and p95 latency per model, with the Pareto optimal set and one
recommendation.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_eval.decorators import Clock, EvaluationSession, Invariants, _as_registry
from agent_eval.invariants.base import InvariantViolation
from agent_eval.schemas import Trajectory

if TYPE_CHECKING:
    from agent_eval.telemetry.otel_exporter import TelemetryEmitter


class ModelPrice(BaseModel):
    """Token prices in USD per million tokens."""

    model_config = ConfigDict(extra="forbid")

    input_cost_per_mtok: float = Field(ge=0)
    output_cost_per_mtok: float = Field(ge=0)


class ModelConfig(BaseModel):
    """One configuration to benchmark. Prices are USD per million tokens.

    ``input_cost_per_mtok`` and ``output_cost_per_mtok`` are the default
    prices. For a configuration that mixes models (a small model for triage,
    a large one for resolution) add ``pricing``, keyed by the ``model`` set on
    each span. Spans whose model is not in the map use the default prices.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    input_cost_per_mtok: float = Field(ge=0)
    output_cost_per_mtok: float = Field(ge=0)
    pricing: dict[str, ModelPrice] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)

    def cost_usd(
        self, input_tokens: float, output_tokens: float, model: str | None = None
    ) -> float:
        price = self.pricing.get(model) if model else None
        input_price = price.input_cost_per_mtok if price else self.input_cost_per_mtok
        output_price = price.output_cost_per_mtok if price else self.output_cost_per_mtok
        return (input_tokens * input_price + output_tokens * output_price) / 1_000_000

    def trajectory_cost_usd(self, trajectory: Trajectory) -> float:
        """Exact cost of a run, priced per span and without double counting."""
        return sum(
            self.cost_usd(input_tokens, output_tokens, span.model)
            for span, input_tokens, output_tokens in trajectory.token_attribution()
        )


class EvalCase(BaseModel):
    """One dataset row.

    ``expect`` is the verdict a correct system produces for this case:
    ``"pass"`` for work the agent should complete cleanly, ``"block"`` for
    adversarial input the engine must stop. For ``"block"`` cases,
    ``expected_violations`` names the invariants that must fire.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    input: Any
    expect: Literal["pass", "block"] = "pass"
    expected_violations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseResult(BaseModel):
    case_id: str
    model: str
    passed: bool
    verdict: Literal["pass", "block"]
    expected: Literal["pass", "block"]
    violations: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class ModelSummary(BaseModel):
    model: str
    cases: int
    passed: int
    pass_rate: float
    total_cost_usd: float
    p95_latency_ms: float
    mean_latency_ms: float = 0.0
    total_tokens: int = 0
    pareto_optimal: bool = False
    recommended: bool = False


class FrontierReport(BaseModel):
    min_pass_rate: float
    summaries: list[ModelSummary]
    results: list[CaseResult] = Field(default_factory=list)
    recommended: str | None = None

    def to_markdown(self) -> str:
        """Recommendation table, best pass rate first."""
        header = (
            "| Model | Pass rate | Passed | Total cost (USD) | p95 latency (ms) | Pareto optimal "
            "| Recommendation |\n"
            "|---|---:|---:|---:|---:|:---:|---|\n"
        )
        rows = []
        ordered = sorted(self.summaries, key=lambda s: (-s.pass_rate, s.total_cost_usd, s.model))
        for s in ordered:
            if s.recommended:
                note = "**recommended**"
            elif not s.pareto_optimal:
                note = "dominated"
            elif s.pass_rate < self.min_pass_rate:
                note = f"below {self.min_pass_rate:.0%} pass floor"
            else:
                note = "viable"
            rows.append(
                f"| {s.model} | {s.pass_rate:.1%} | {s.passed}/{s.cases} | "
                f"{s.total_cost_usd:.6f} | {s.p95_latency_ms:.0f} | "
                f"{'yes' if s.pareto_optimal else 'no'} | {note} |"
            )
        return header + "\n".join(rows)


# Sync or async: may return the output or an awaitable of it.
AgentRunner = Callable[[EvalCase, ModelConfig, EvaluationSession], Any]


def load_dataset(path: str | Path) -> list[EvalCase]:
    """Load a JSONL dataset. Blank lines and ``//`` comment lines are skipped."""
    cases: list[EvalCase] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            cases.append(EvalCase.model_validate(json.loads(stripped)))
        except Exception as exc:
            raise ValueError(f"{path}:{number}: invalid case: {exc}") from exc
    ids = [case.id for case in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate case ids: {', '.join(duplicates)}")
    return cases


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest rank percentile. ``percentile(xs, 95)`` is the p95."""
    if not values:
        return 0.0
    if not 0 < pct <= 100:
        raise ValueError("pct must be in (0, 100]")
    ordered = sorted(values)
    rank = math.ceil(pct / 100 * len(ordered))
    return ordered[max(rank, 1) - 1]


def dominates(a: ModelSummary, b: ModelSummary) -> bool:
    """True when ``a`` is at least as good as ``b`` everywhere and better somewhere.

    Objectives: maximise pass rate, minimise total cost, minimise p95 latency.
    """
    no_worse = (
        a.pass_rate >= b.pass_rate
        and a.total_cost_usd <= b.total_cost_usd
        and a.p95_latency_ms <= b.p95_latency_ms
    )
    better = (
        a.pass_rate > b.pass_rate
        or a.total_cost_usd < b.total_cost_usd
        or a.p95_latency_ms < b.p95_latency_ms
    )
    return no_worse and better


def pareto_frontier(summaries: Iterable[ModelSummary]) -> list[ModelSummary]:
    """The non dominated subset, in input order."""
    items = list(summaries)
    return [s for s in items if not any(dominates(other, s) for other in items if other is not s)]


def recommend(frontier: Sequence[ModelSummary], min_pass_rate: float) -> ModelSummary | None:
    """Cheapest frontier model that clears the pass floor.

    If none clears it, fall back to the highest pass rate (then cheapest), so
    the table always names the least bad option.
    """
    if not frontier:
        return None
    viable = [s for s in frontier if s.pass_rate >= min_pass_rate]
    if viable:
        return min(
            viable, key=lambda s: (s.total_cost_usd, -s.pass_rate, s.p95_latency_ms, s.model)
        )
    return min(frontier, key=lambda s: (-s.pass_rate, s.total_cost_usd, s.p95_latency_ms, s.model))


def summarize(
    results: Iterable[CaseResult], min_pass_rate: float = 0.9
) -> FrontierReport:
    """Aggregate case results per model and mark the frontier and recommendation."""
    all_results = list(results)
    by_model: dict[str, list[CaseResult]] = {}
    for result in all_results:
        by_model.setdefault(result.model, []).append(result)

    summaries: list[ModelSummary] = []
    for model, rows in by_model.items():
        latencies = [r.latency_ms for r in rows]
        passed = sum(1 for r in rows if r.passed)
        summaries.append(
            ModelSummary(
                model=model,
                cases=len(rows),
                passed=passed,
                pass_rate=passed / len(rows),
                total_cost_usd=round(sum(r.cost_usd for r in rows), 9),
                p95_latency_ms=percentile(latencies, 95),
                mean_latency_ms=round(sum(latencies) / len(latencies), 3),
                total_tokens=sum(r.input_tokens + r.output_tokens for r in rows),
            )
        )

    frontier = pareto_frontier(summaries)
    for summary in frontier:
        summary.pareto_optimal = True
    pick = recommend(frontier, min_pass_rate)
    if pick is not None:
        pick.recommended = True
    return FrontierReport(
        min_pass_rate=min_pass_rate,
        summaries=summaries,
        results=all_results,
        recommended=pick.model if pick else None,
    )


class FrontierRunner:
    """Replays a dataset across model configurations.

    ``agent`` is any callable ``(case, model, session) -> output``, sync or
    ``async def``. It drives the real workflow with the given configuration
    and records spans on the session. The runner sets the input, captures the
    output, and scores.

    A case passes when the verdict equals ``case.expect`` and every name in
    ``case.expected_violations`` actually fired. An agent exception is a
    blocked run (fail closed).

    ``max_concurrency`` runs that many cases at once: worker threads in
    :meth:`run`, tasks on the caller's event loop in :meth:`arun`. Results
    keep dataset order either way. Use :meth:`arun` when the agent shares
    loop bound clients (one ``httpx.AsyncClient``, one SDK client).
    """

    def __init__(
        self,
        agent: AgentRunner,
        *,
        invariants: Invariants = None,
        min_pass_rate: float = 0.9,
        clock_factory: Callable[[], Clock] | None = None,
        emitter: TelemetryEmitter | None = None,
        max_concurrency: int = 1,
    ) -> None:
        if not 0.0 <= min_pass_rate <= 1.0:
            raise ValueError("min_pass_rate must be between 0 and 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.agent = agent
        self.registry = _as_registry(invariants)
        self.min_pass_rate = min_pass_rate
        self.clock_factory = clock_factory
        self.emitter = emitter
        self.max_concurrency = max_concurrency

    # ---- one case ---------------------------------------------------------------

    def _session(self, case: EvalCase, model: ModelConfig) -> EvaluationSession:
        return EvaluationSession(
            case.id,
            invariants=self.registry,
            enforce=True,
            emitter=self.emitter,
            clock=self.clock_factory() if self.clock_factory else None,
            model=model.name,
        )

    @staticmethod
    def _score(
        case: EvalCase, model: ModelConfig, session: EvaluationSession, error: str | None
    ) -> CaseResult:
        report = session.report
        if report is None:  # defensive: no verdict means blocked
            verdict: Literal["pass", "block"] = "block"
            violations: list[str] = []
            metrics: dict[str, float] = session.trajectory.metrics()
        else:
            verdict = "pass" if report.passed else "block"
            violations = report.violated_names
            metrics = report.metrics
        matched = verdict == case.expect and set(case.expected_violations) <= set(violations)
        return CaseResult(
            case_id=case.id,
            model=model.name,
            passed=matched,
            verdict=verdict,
            expected=case.expect,
            violations=violations,
            latency_ms=float(metrics.get("latency_ms", 0.0)),
            input_tokens=int(metrics.get("input_tokens", 0)),
            output_tokens=int(metrics.get("output_tokens", 0)),
            cost_usd=model.trajectory_cost_usd(session.trajectory),
            error=error,
        )

    def run_case(self, case: EvalCase, model: ModelConfig) -> CaseResult:
        """Run one case synchronously. An async agent gets its own event loop."""
        try:
            asyncio.get_running_loop()
            in_event_loop = True
        except RuntimeError:
            in_event_loop = False

        session = self._session(case, model)
        error: str | None = None
        misuse = False
        try:
            with session:
                session.set_input(case.input)
                output = self.agent(case, model, session)
                if inspect.isawaitable(output):
                    if in_event_loop:
                        misuse = True
                        if inspect.iscoroutine(output):
                            output.close()
                        raise RuntimeError("async agent inside a running event loop")
                    output = asyncio.run(_await(output))
                session.set_output(output)
        except InvariantViolation:
            pass
        except Exception as exc:  # noqa: BLE001 - recorded, the run counts as blocked
            error = f"{type(exc).__name__}: {exc}"
        if misuse:
            raise RuntimeError(
                "run() and run_case() cannot drive an async agent from inside a running "
                "event loop. Use 'await runner.arun(...)' instead."
            )
        return self._score(case, model, session, error)

    async def arun_case(self, case: EvalCase, model: ModelConfig) -> CaseResult:
        """Run one case on the current event loop. A sync agent blocks the loop."""
        session = self._session(case, model)
        error: str | None = None
        try:
            with session:
                session.set_input(case.input)
                output = self.agent(case, model, session)
                if inspect.isawaitable(output):
                    output = await output
                session.set_output(output)
        except InvariantViolation:
            pass
        except Exception as exc:  # noqa: BLE001 - recorded, the run counts as blocked
            error = f"{type(exc).__name__}: {exc}"
        return self._score(case, model, session, error)

    # ---- whole dataset ----------------------------------------------------------

    @staticmethod
    def _plan(
        cases: Sequence[EvalCase], models: Sequence[ModelConfig]
    ) -> list[tuple[EvalCase, ModelConfig]]:
        if not cases:
            raise ValueError("dataset is empty")
        if not models:
            raise ValueError("no model configurations given")
        names = [m.name for m in models]
        if len(set(names)) != len(names):
            raise ValueError("model configuration names must be unique")
        return [(case, model) for model in models for case in cases]

    def run(self, cases: Sequence[EvalCase], models: Sequence[ModelConfig]) -> FrontierReport:
        plan = self._plan(cases, models)
        if self.max_concurrency == 1:
            results = [self.run_case(case, model) for case, model in plan]
        else:
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
                results = list(pool.map(lambda pair: self.run_case(*pair), plan))
        return summarize(results, self.min_pass_rate)

    async def arun(
        self, cases: Sequence[EvalCase], models: Sequence[ModelConfig]
    ) -> FrontierReport:
        plan = self._plan(cases, models)
        gate = asyncio.Semaphore(self.max_concurrency)

        async def guarded(case: EvalCase, model: ModelConfig) -> CaseResult:
            async with gate:
                return await self.arun_case(case, model)

        results = await asyncio.gather(*(guarded(case, model) for case, model in plan))
        return summarize(list(results), self.min_pass_rate)


async def _await(awaitable: Any) -> Any:
    return await awaitable
