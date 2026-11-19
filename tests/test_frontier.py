"""Pareto frontier, percentile, recommendation and the replay runner."""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_eval import SimClock, default_registry
from agent_eval.benchmark import (
    CaseResult,
    EvalCase,
    FrontierRunner,
    ModelConfig,
    ModelPrice,
    ModelSummary,
    dominates,
    load_dataset,
    pareto_frontier,
    percentile,
    recommend,
    summarize,
)


def summary(model: str, pass_rate: float, cost: float, p95: float) -> ModelSummary:
    return ModelSummary(
        model=model,
        cases=10,
        passed=round(pass_rate * 10),
        pass_rate=pass_rate,
        total_cost_usd=cost,
        p95_latency_ms=p95,
    )


# ---- percentile -----------------------------------------------------------------

def test_percentile_nearest_rank():
    values = list(range(1, 101))
    assert percentile(values, 95) == 95
    assert percentile(values, 100) == 100
    assert percentile([10.0, 20.0], 95) == 20.0
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 95) == 0.0
    with pytest.raises(ValueError):
        percentile([1.0], 0)


# ---- dominance and frontier -----------------------------------------------------

def test_dominates_needs_strict_improvement():
    a = summary("a", 0.9, 1.0, 100)
    assert not dominates(a, summary("twin", 0.9, 1.0, 100))
    assert dominates(a, summary("pricier", 0.9, 2.0, 100))
    assert not dominates(a, summary("tradeoff", 1.0, 5.0, 100))


def test_pareto_frontier_drops_dominated_models():
    frontier_xl = summary("frontier-xl", 1.0, 10.0, 3000)
    balanced = summary("balanced", 0.9, 2.0, 1500)
    legacy = summary("legacy", 0.9, 3.0, 2000)  # dominated by balanced
    distilled = summary("distilled", 0.5, 0.2, 600)
    frontier = pareto_frontier([frontier_xl, balanced, legacy, distilled])
    assert [s.model for s in frontier] == ["frontier-xl", "balanced", "distilled"]


def test_latency_alone_can_keep_a_model_on_the_frontier():
    cheap_slow = summary("cheap-slow", 0.9, 1.0, 5000)
    pricey_fast = summary("pricey-fast", 0.9, 2.0, 500)
    assert len(pareto_frontier([cheap_slow, pricey_fast])) == 2


def test_identical_models_all_stay_on_the_frontier():
    twins = [summary("a", 0.8, 1.0, 100), summary("b", 0.8, 1.0, 100)]
    assert len(pareto_frontier(twins)) == 2


def test_recommend_cheapest_above_floor_else_best_pass_rate():
    frontier = [
        summary("frontier-xl", 1.0, 10.0, 3000),
        summary("balanced", 0.9, 2.0, 1500),
        summary("distilled", 0.5, 0.2, 600),
    ]
    assert recommend(frontier, 0.9).model == "balanced"
    assert recommend(frontier, 0.95).model == "frontier-xl"
    assert recommend(frontier[1:], 0.99).model == "balanced"  # nothing clears the floor
    assert recommend([], 0.9) is None


def test_summarize_aggregates_and_flags():
    def row(model: str, case: str, passed: bool, latency: float, cost: float) -> CaseResult:
        return CaseResult(
            case_id=case,
            model=model,
            passed=passed,
            verdict="pass" if passed else "block",
            expected="pass",
            latency_ms=latency,
            cost_usd=cost,
        )

    report = summarize(
        [
            row("big", "c1", True, 900, 0.5),
            row("big", "c2", True, 1100, 0.5),
            row("small", "c1", True, 100, 0.01),
            row("small", "c2", False, 120, 0.01),
        ],
        min_pass_rate=0.9,
    )
    by_model = {s.model: s for s in report.summaries}
    assert by_model["big"].pass_rate == 1.0
    assert by_model["big"].total_cost_usd == 1.0
    assert by_model["big"].p95_latency_ms == 1100
    assert by_model["small"].pass_rate == 0.5
    assert all(s.pareto_optimal for s in report.summaries)
    assert report.recommended == "big"
    table = report.to_markdown()
    assert "**recommended**" in table and "below 90% pass floor" in table


# ---- runner ---------------------------------------------------------------------

BIG = ModelConfig(name="big", input_cost_per_mtok=10.0, output_cost_per_mtok=30.0,
                  params={"accurate": True, "ms": 1000})
SMALL = ModelConfig(name="small", input_cost_per_mtok=1.0, output_cost_per_mtok=3.0,
                    params={"accurate": False, "ms": 100})


def toy_agent(case, model, session):
    """Looks up a price with a tool; the inaccurate model misquotes it."""
    with session.agent("pricing") as step:
        session.clock.advance(model.params["ms"])
        step.set_tokens(1000, 100)
        with session.tool("price.lookup", input={"sku": case.input["sku"]}) as call:
            call.set_output({"price": 20.0})
        price = 20.0 if model.params["accurate"] or not case.metadata.get("hard") else 25.0
        return f"The price is ${price:.2f}."


CASES = [
    EvalCase(id="easy", input={"sku": "A"}),
    EvalCase(id="hard", input={"sku": "B"}, metadata={"hard": True}),
    EvalCase(
        id="pii-probe",
        input={"sku": "C", "note": "mail me at jane.doe@example.com"},
        expect="block",
        expected_violations=["no_pii"],
    ),
]


def test_runner_scores_cost_latency_and_expectations():
    runner = FrontierRunner(
        toy_agent, invariants=default_registry(), min_pass_rate=0.9, clock_factory=SimClock
    )
    report = runner.run(CASES, [BIG, SMALL])
    by_model = {s.model: s for s in report.summaries}

    assert by_model["big"].passed == 3
    assert by_model["small"].passed == 2
    assert report.recommended == "big"

    # Two cases reach the model (the PII probe is blocked at the input gate).
    assert by_model["big"].total_cost_usd == pytest.approx(2 * (1000 * 10 + 100 * 30) / 1e6)
    assert by_model["small"].total_cost_usd == pytest.approx(2 * (1000 * 1 + 100 * 3) / 1e6)
    assert by_model["big"].p95_latency_ms == 1000
    assert by_model["small"].p95_latency_ms == 100

    results = {(r.model, r.case_id): r for r in report.results}
    assert results[("small", "hard")].violations == ["numeric_grounding"]
    assert results[("big", "pii-probe")].verdict == "block"
    assert results[("big", "pii-probe")].passed is True
    assert results[("big", "pii-probe")].input_tokens == 0


def test_runner_is_deterministic():
    def run() -> str:
        runner = FrontierRunner(toy_agent, invariants=default_registry(), clock_factory=SimClock)
        report = runner.run(CASES, [BIG, SMALL])
        return json.dumps([s.model_dump() for s in report.summaries], sort_keys=True)

    assert run() == run()


def test_agent_exception_counts_as_blocked():
    def broken(case, model, session):
        raise RuntimeError("provider outage")

    runner = FrontierRunner(broken, invariants=default_registry(), clock_factory=SimClock)
    result = runner.run_case(CASES[0], BIG)
    assert result.verdict == "block"
    assert result.passed is False
    assert result.error == "RuntimeError: provider outage"
    assert "agent_completed_without_error" in result.violations


def test_expected_violation_must_actually_fire():
    case = EvalCase(
        id="wrong-reason",
        input={"sku": "C", "note": "mail me at jane.doe@example.com"},
        expect="block",
        expected_violations=["pci_compliant"],
    )
    runner = FrontierRunner(toy_agent, invariants=default_registry(), clock_factory=SimClock)
    assert runner.run_case(case, BIG).passed is False


def test_runner_rejects_empty_or_duplicate_inputs():
    runner = FrontierRunner(toy_agent, invariants=default_registry())
    with pytest.raises(ValueError):
        runner.run([], [BIG])
    with pytest.raises(ValueError):
        runner.run(CASES, [])
    with pytest.raises(ValueError):
        runner.run(CASES, [BIG, BIG])


def test_load_dataset(tmp_path):
    path = tmp_path / "suite.jsonl"
    path.write_text(
        '// comment\n{"id": "a", "input": "x"}\n\n'
        '{"id": "b", "input": "y", "expect": "block", "expected_violations": ["no_pii"]}\n'
    )
    cases = load_dataset(path)
    assert [c.id for c in cases] == ["a", "b"]
    assert cases[1].expect == "block"

    path.write_text('{"id": "a", "input": "x"}\n{"id": "a", "input": "y"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(path)

    path.write_text('{"id": "a"}\n')
    with pytest.raises(ValueError, match="suite.jsonl:1"):
        load_dataset(path)


# ---- Review fixes: async agents, concurrency, per span pricing ------------------

async def async_toy_agent(case, model, session):
    with session.agent("pricing") as step:
        await asyncio.sleep(0)
        session.clock.advance(model.params["ms"])
        step.set_tokens(1000, 100)
        with session.tool("price.lookup", input={"sku": case.input["sku"]}) as call:
            call.set_output({"price": 20.0})
        price = 20.0 if model.params["accurate"] or not case.metadata.get("hard") else 25.0
        return f"The price is ${price:.2f}."


def _summaries(report) -> str:
    return json.dumps([s.model_dump() for s in report.summaries], sort_keys=True)


def test_async_agent_and_concurrency_match_the_sequential_run():
    def runner(agent, concurrency):
        return FrontierRunner(
            agent, invariants=default_registry(), clock_factory=SimClock,
            max_concurrency=concurrency,
        )

    baseline = runner(toy_agent, 1).run(CASES, [BIG, SMALL])
    threaded = runner(toy_agent, 4).run(CASES, [BIG, SMALL])
    async_in_sync = runner(async_toy_agent, 4).run(CASES, [BIG, SMALL])
    on_loop = asyncio.run(runner(async_toy_agent, 4).arun(CASES, [BIG, SMALL]))
    sync_on_loop = asyncio.run(runner(toy_agent, 2).arun(CASES, [BIG, SMALL]))

    expected = _summaries(baseline)
    for report in (threaded, async_in_sync, on_loop, sync_on_loop):
        assert _summaries(report) == expected
        assert [(r.model, r.case_id) for r in report.results] == [
            (r.model, r.case_id) for r in baseline.results
        ]


def test_sync_run_inside_event_loop_with_async_agent_is_refused():
    runner = FrontierRunner(async_toy_agent, invariants=default_registry(), clock_factory=SimClock)

    async def misuse():
        return runner.run(CASES, [BIG])

    with pytest.raises(RuntimeError, match="arun"):
        asyncio.run(misuse())


def test_max_concurrency_validated():
    with pytest.raises(ValueError):
        FrontierRunner(toy_agent, max_concurrency=0)


def test_mixed_model_pricing_per_span():
    config = ModelConfig(
        name="mixed",
        input_cost_per_mtok=100.0,   # default, used for unknown models
        output_cost_per_mtok=100.0,
        pricing={
            "small": ModelPrice(input_cost_per_mtok=1.0, output_cost_per_mtok=2.0),
            "large": ModelPrice(input_cost_per_mtok=10.0, output_cost_per_mtok=20.0),
        },
    )

    def mixed_agent(case, model, session):
        with session.agent("triage", model="small") as step:
            step.set_tokens(1_000_000, 1_000_000)
        with session.agent("resolution", model="large") as step:
            step.set_tokens(1_000_000, 1_000_000)
        with session.agent("audit", model="unknown") as step:
            step.set_tokens(1_000_000, 0)
        return "done"

    runner = FrontierRunner(mixed_agent, invariants=default_registry(), clock_factory=SimClock)
    result = runner.run_case(EvalCase(id="c", input="x"), config)
    assert result.cost_usd == pytest.approx(1 + 2 + 10 + 20 + 100)


def test_rolled_up_usage_is_priced_once():
    def rollup_agent(case, model, session):
        with session.agent("a") as step:
            with session.llm("m") as call:
                call.set_tokens(1_000_000, 0)
            step.set_tokens(1_000_000, 0)  # framework repeats the child usage
        return "done"

    config = ModelConfig(name="m", input_cost_per_mtok=3.0, output_cost_per_mtok=0.0)
    runner = FrontierRunner(rollup_agent, invariants=default_registry(), clock_factory=SimClock)
    result = runner.run_case(EvalCase(id="c", input="x"), config)
    assert result.input_tokens == 1_000_000
    assert result.cost_usd == pytest.approx(3.0)
