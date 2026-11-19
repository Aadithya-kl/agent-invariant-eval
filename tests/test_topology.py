"""Cycle detection, hop limits, DAG validation, context bloat, swallowed failures, SLA."""

from __future__ import annotations

import pytest

from agent_eval import Span, SpanKind, SpanStatus, Stage, Trajectory
from agent_eval.invariants import (
    assert_no_context_bloat,
    assert_no_routing_loops,
    assert_no_swallowed_tool_failures,
    assert_sla_bounds,
    assert_valid_dag,
    find_cycle,
)

MS = 1_000_000


def agent(name: str, at: int, *, input_tokens: int = 0, span_id: str | None = None,
          parent_id: str | None = None) -> Span:
    kwargs = {"span_id": span_id} if span_id else {}
    return Span(
        kind=SpanKind.AGENT,
        name=f"invoke_agent {name}",
        agent_name=name,
        input_tokens=input_tokens,
        start_ns=at * MS,
        end_ns=(at + 1) * MS,
        parent_id=parent_id,
        **kwargs,
    )


def tool(name: str, at: int, *, ok: bool = True, output=None, handled: bool = False) -> Span:
    return Span(
        kind=SpanKind.TOOL,
        name=f"execute_tool {name}",
        tool_name=name,
        output=output,
        start_ns=at * MS,
        end_ns=(at + 1) * MS,
        status=SpanStatus.OK if ok else SpanStatus.ERROR,
        error=None if ok else "TimeoutError: upstream timed out",
        attributes={"error.handled": True} if handled else {},
    )


def route(*names: str) -> Trajectory:
    return Trajectory(spans=[agent(name, index + 1) for index, name in enumerate(names)])


# ---- find_cycle -----------------------------------------------------------------

def test_find_cycle_on_acyclic_graph():
    assert find_cycle([("a", "b"), ("b", "c"), ("a", "c")]) is None
    assert find_cycle([]) is None


def test_find_cycle_returns_closed_path():
    cycle = find_cycle([("a", "b"), ("b", "c"), ("c", "a")])
    assert cycle is not None
    assert cycle[0] == cycle[-1]
    assert set(cycle) == {"a", "b", "c"}


def test_find_cycle_self_loop_and_deep_chain():
    assert find_cycle([("a", "a")]) == ["a", "a"]
    chain = [(i, i + 1) for i in range(5000)]
    assert find_cycle(chain) is None
    assert find_cycle([*chain, (5000, 4000)]) is not None


# ---- routing loops and hop limits -----------------------------------------------

def test_linear_route_passes():
    trajectory = route("triage", "ledger_lookup", "resolution")
    assert trajectory.agent_sequence() == ["triage", "ledger_lookup", "resolution"]
    result = assert_no_routing_loops(max_hops=4).evaluate(trajectory)
    assert result.passed is True
    assert result.metrics["hops"] == 2
    assert result.metrics["cyclic"] == 0


def test_consecutive_spans_of_one_agent_are_one_visit():
    trajectory = route("triage", "triage", "triage", "resolution")
    assert trajectory.agent_sequence() == ["triage", "resolution"]
    assert assert_no_routing_loops(max_hops=1, max_visits_per_agent=1).evaluate(trajectory).passed


def test_single_return_to_supervisor_is_allowed_by_default():
    result = assert_no_routing_loops(max_hops=4).evaluate(route("a", "b", "a"))
    assert result.passed is True
    assert result.metrics["cyclic"] == 1


def test_ping_pong_loop_detected_with_cycle_evidence():
    result = assert_no_routing_loops(max_hops=10).evaluate(route("a", "b", "a", "b"))
    assert result.passed is False
    assert "a -> b x2" in result.message
    assert "cycle: a -> b -> a" in result.evidence


def test_supervisor_hub_and_spoke_is_not_a_loop():
    hub = route("supervisor", "worker_a", "supervisor", "worker_b", "supervisor", "worker_c")
    result = assert_no_routing_loops(max_hops=8).evaluate(hub)
    assert result.passed is True
    assert result.metrics["max_visits"] == 3
    assert result.metrics["max_edge_repeats"] == 1

    redelegation = route("supervisor", "worker_a", "supervisor", "worker_a")
    assert assert_no_routing_loops(max_hops=8).evaluate(redelegation).passed is False


def test_bounded_review_loop_allowed_by_raising_edge_repeats():
    review = route("writer", "reviewer", "writer", "reviewer", "writer")
    assert assert_no_routing_loops(max_hops=8).evaluate(review).passed is False
    assert assert_no_routing_loops(max_hops=8, max_edge_repeats=2).evaluate(review).passed is True


def test_strict_mode_rejects_any_revisit():
    strict = assert_no_routing_loops(max_hops=10, max_visits_per_agent=1)
    assert strict.evaluate(route("a", "b", "a")).passed is False


def test_hop_limit_enforced_without_any_cycle():
    trajectory = route("a", "b", "c", "d", "e")
    assert assert_no_routing_loops(max_hops=4).evaluate(trajectory).passed is True
    result = assert_no_routing_loops(max_hops=3).evaluate(trajectory)
    assert result.passed is False
    assert "4 hops exceeds max_hops=3" in result.message


def test_empty_trajectory_has_no_loops():
    assert assert_no_routing_loops(max_hops=0).evaluate(Trajectory()).passed is True


# ---- span DAG -------------------------------------------------------------------

def test_valid_dag_and_topological_order():
    trajectory = Trajectory(
        spans=[
            agent("child", 2, span_id="c", parent_id="p"),
            agent("parent", 1, span_id="p"),
        ]
    )
    assert assert_valid_dag().evaluate(trajectory).passed is True
    assert [s.span_id for s in trajectory.topological_spans()] == ["p", "c"]


def test_dag_rejects_unknown_parent_duplicate_ids_and_cycles():
    orphan = Trajectory(spans=[agent("a", 1, span_id="a", parent_id="ghost")])
    assert assert_valid_dag().evaluate(orphan).passed is False

    duplicate = Trajectory(spans=[agent("a", 1, span_id="x"), agent("b", 2, span_id="x")])
    assert assert_valid_dag().evaluate(duplicate).passed is False

    cyclic = Trajectory(
        spans=[agent("a", 1, span_id="a", parent_id="b"), agent("b", 2, span_id="b", parent_id="a")]
    )
    result = assert_valid_dag().evaluate(cyclic)
    assert result.passed is False
    with pytest.raises(ValueError):
        cyclic.topological_spans()


# ---- context bloat --------------------------------------------------------------

def test_moderate_growth_passes():
    trajectory = Trajectory(
        spans=[agent("a", 1, input_tokens=350), agent("b", 2, input_tokens=650)]
    )
    result = assert_no_context_bloat().evaluate(trajectory)
    assert result.passed is True
    assert result.metrics["cumulative_input_tokens"] == 1000


def test_runaway_growth_fails():
    trajectory = Trajectory(
        spans=[
            agent("a", 1, input_tokens=400),
            agent("b", 2, input_tokens=2400),
            agent("c", 3, input_tokens=9600),
        ]
    )
    result = assert_no_context_bloat(max_growth_factor=4.0).evaluate(trajectory)
    assert result.passed is False
    assert result.metrics["growth_factor"] == 19.2  # 9600 over the 500 token floor
    unfloored = assert_no_context_bloat(min_baseline_tokens=1).evaluate(trajectory)
    assert unfloored.metrics["growth_factor"] == 24.0
    assert result.evidence == ["input tokens per step: a:400 -> b:2400 -> c:9600"]


def test_cumulative_input_token_cap():
    trajectory = Trajectory(
        spans=[agent("a", 1, input_tokens=600), agent("b", 2, input_tokens=600)]
    )
    assert assert_no_context_bloat(max_cumulative_input_tokens=1200).evaluate(trajectory).passed
    capped = assert_no_context_bloat(max_cumulative_input_tokens=1000).evaluate(trajectory)
    assert capped.passed is False


# ---- swallowed tool failures ----------------------------------------------------

def test_swallowed_failure_detected():
    trajectory = Trajectory(
        spans=[tool("ledger", 1, ok=False)],
        final_output="Good news, your refund is on its way.",
    )
    result = assert_no_swallowed_tool_failures().evaluate(trajectory)
    assert result.passed is False
    assert result.span_ids == [trajectory.spans[0].span_id]


def test_error_payload_with_ok_status_counts_as_failure():
    trajectory = Trajectory(
        spans=[tool("ledger", 1, output={"error": "not found"})],
        final_output="Your refund is on its way.",
    )
    assert assert_no_swallowed_tool_failures().evaluate(trajectory).passed is False


def test_retry_acknowledgement_handled_flag_and_propagation_all_pass():
    invariant = assert_no_swallowed_tool_failures()
    confident = "Your refund is on its way."

    retried = Trajectory(
        spans=[tool("ledger", 1, ok=False), tool("ledger", 2, output={"amount": 5})],
        final_output=confident,
    )
    assert invariant.evaluate(retried).passed is True

    other_tool = Trajectory(
        spans=[tool("ledger", 1, ok=False), tool("crm", 2, output={})], final_output=confident
    )
    assert invariant.evaluate(other_tool).passed is False

    acknowledged = Trajectory(
        spans=[tool("ledger", 1, ok=False)],
        final_output="We were unable to reach the ledger. Please try again later.",
    )
    assert invariant.evaluate(acknowledged).passed is True

    handled = Trajectory(spans=[tool("ledger", 1, ok=False, handled=True)], final_output=confident)
    assert invariant.evaluate(handled).passed is True

    propagated = Trajectory(spans=[tool("ledger", 1, ok=False)], error="TimeoutError: upstream")
    assert invariant.evaluate(propagated).passed is True


# ---- SLA bounds -----------------------------------------------------------------

def test_sla_latency_and_token_bounds():
    trajectory = Trajectory(
        start_ns=1,
        end_ns=1 + 500 * MS,
        spans=[agent("a", 1, input_tokens=900)],
    )
    assert assert_sla_bounds(max_latency_ms=500, max_tokens=900).evaluate(trajectory).passed
    slow = assert_sla_bounds(max_latency_ms=499).evaluate(trajectory)
    assert slow.passed is False and "latency" in slow.message
    heavy = assert_sla_bounds(max_tokens=899).evaluate(trajectory, Stage.SPAN)
    assert heavy.passed is False and "tokens" in heavy.message


def test_sla_requires_a_bound():
    with pytest.raises(ValueError):
        assert_sla_bounds()


# ---- Review fixes ---------------------------------------------------------------

def test_small_router_prompt_does_not_trip_context_bloat():
    trajectory = Trajectory(
        spans=[agent("router", 1, input_tokens=50), agent("worker", 2, input_tokens=300)]
    )
    result = assert_no_context_bloat().evaluate(trajectory)
    assert result.passed is True
    assert result.metrics["baseline_tokens"] == 500
    assert assert_no_context_bloat(min_baseline_tokens=1).evaluate(trajectory).passed is False


def test_apology_is_not_an_acknowledgement():
    trajectory = Trajectory(
        spans=[tool("ledger", 1, ok=False)],
        final_output="Sorry for the billing error! Your refund is approved.",
    )
    assert assert_no_swallowed_tool_failures().evaluate(trajectory).passed is False


def test_acknowledgement_heuristic_can_be_disabled():
    trajectory = Trajectory(
        spans=[tool("ledger", 1, ok=False)],
        final_output="We were unable to reach the ledger. Please try again later.",
    )
    assert assert_no_swallowed_tool_failures().evaluate(trajectory).passed is True
    strict = assert_no_swallowed_tool_failures(allow_acknowledgement=False)
    assert strict.evaluate(trajectory).passed is False


def llm(at: int, parent: str, tokens_in: int, tokens_out: int = 0) -> Span:
    return Span(
        kind=SpanKind.LLM, name="chat m", model="m", parent_id=parent,
        input_tokens=tokens_in, output_tokens=tokens_out, start_ns=at * MS, end_ns=(at + 1) * MS,
    )


def test_rolled_up_agent_tokens_are_not_double_counted():
    parent = agent("a", 1, input_tokens=700, span_id="p")
    trajectory = Trajectory(spans=[parent, llm(2, "p", 300), llm(3, "p", 400)])
    assert trajectory.total_input_tokens == 700
    assert [s.kind for s, _, _ in trajectory.token_attribution()] == [SpanKind.LLM, SpanKind.LLM]
    assert assert_sla_bounds(max_tokens=700).evaluate(trajectory).passed is True


def test_roll_up_with_extra_parent_usage_keeps_the_remainder():
    parent = agent("a", 1, input_tokens=1000, span_id="p")
    trajectory = Trajectory(spans=[parent, llm(2, "p", 300), llm(3, "p", 400)])
    assert trajectory.total_input_tokens == 1000
    remainder = {s.span_id: tokens for s, tokens, _ in trajectory.token_attribution()}["p"]
    assert remainder == 300


def test_parent_smaller_than_children_is_counted_on_top():
    parent = agent("a", 1, input_tokens=100, span_id="p")
    trajectory = Trajectory(spans=[parent, llm(2, "p", 300), llm(3, "p", 400)])
    assert trajectory.total_input_tokens == 800

    mixed = Span(
        kind=SpanKind.AGENT, name="invoke_agent a", agent_name="a",
        input_tokens=1000, output_tokens=10, start_ns=MS, end_ns=2 * MS, span_id="m",
    )
    mixed_traj = Trajectory(spans=[mixed, llm(2, "m", 100, 50)])
    assert mixed_traj.total_input_tokens == 1100
    assert mixed_traj.total_output_tokens == 60


def test_invalid_dag_counts_every_span_in_full():
    cyclic = Trajectory(
        spans=[
            agent("a", 1, input_tokens=10, span_id="a", parent_id="b"),
            agent("b", 2, input_tokens=10, span_id="b", parent_id="a"),
        ]
    )
    assert cyclic.total_input_tokens == 20


def test_context_bloat_ignores_rolled_up_parent():
    parent = agent("a", 1, input_tokens=5200, span_id="p")
    trajectory = Trajectory(spans=[parent, llm(2, "p", 2600), llm(3, "p", 2600)])
    result = assert_no_context_bloat().evaluate(trajectory)
    assert result.passed is True
    assert result.metrics["cumulative_input_tokens"] == 5200
