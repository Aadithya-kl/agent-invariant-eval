"""Topology and trajectory analyzers.

These inspect the shape of an execution rather than its content: the span DAG,
agent to agent routing, token growth across handoffs, tool failures that never
surfaced, and latency or token budgets.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping
from typing import Any, ClassVar, TypeVar

from agent_eval.invariants.base import BaseInvariant
from agent_eval.invariants.compliance import flatten_text
from agent_eval.schemas import (
    AssertionResult,
    Severity,
    Span,
    SpanKind,
    SpanStatus,
    Stage,
    Trajectory,
)

_RUNTIME_STAGES: frozenset[Stage] = frozenset({Stage.SPAN, Stage.OUTPUT})
_T = TypeVar("_T", bound=Hashable)


def find_cycle(edges: Iterable[tuple[_T, _T]]) -> list[_T] | None:
    """Return one cycle in a directed graph as ``[a, b, ..., a]``, or ``None``.

    Iterative three colour depth first search, so deep graphs do not hit the
    recursion limit.
    """
    graph: dict[_T, list[_T]] = {}
    for source, target in edges:
        graph.setdefault(source, [])
        graph.setdefault(target, [])
        if target not in graph[source]:
            graph[source].append(target)

    white, grey, black = 0, 1, 2
    colour: dict[_T, int] = dict.fromkeys(graph, white)
    for root in graph:
        if colour[root] != white:
            continue
        path: list[_T] = [root]
        stack = [iter(graph[root])]
        colour[root] = grey
        while stack:
            advanced = False
            for nxt in stack[-1]:
                if colour[nxt] == grey:
                    return path[path.index(nxt) :] + [nxt]
                if colour[nxt] == white:
                    colour[nxt] = grey
                    path.append(nxt)
                    stack.append(iter(graph[nxt]))
                    advanced = True
                    break
            if not advanced:
                colour[path.pop()] = black
                stack.pop()
    return None


class ValidSpanDAG(BaseInvariant):
    """Span parent links form a DAG: unique ids, known parents, no cycles."""

    name: ClassVar[str] = "valid_span_dag"
    stages: ClassVar[frozenset[Stage]] = _RUNTIME_STAGES

    def check(self, trajectory: Trajectory, stage: Stage) -> AssertionResult:
        errors = trajectory.dag_errors()
        if errors:
            return self.fail(stage, "span graph is not a valid DAG", evidence=errors)
        return self.ok(stage, "span graph is a valid DAG", spans=float(len(trajectory.spans)))


class NoRoutingLoops(BaseInvariant):
    """Bound agent to agent routing.

    Fails when:

    * the number of handoffs exceeds ``max_hops``, or
    * the same directed handoff (A -> B) is taken more than
      ``max_edge_repeats`` times, or
    * ``max_visits_per_agent`` is set and an agent is entered more often.

    Counting repeated edges rather than node visits keeps hub and spoke
    routing legal: a supervisor that delegates to three different workers
    (S -> A -> S -> B -> S -> C) repeats no edge, while a ping pong
    (S -> A -> S -> A) repeats S -> A and is blocked. Raise
    ``max_edge_repeats`` for workflows with a bounded review loop
    (writer -> reviewer -> writer -> reviewer).
    """

    name: ClassVar[str] = "no_routing_loops"
    stages: ClassVar[frozenset[Stage]] = _RUNTIME_STAGES

    def __init__(
        self,
        *,
        max_hops: int = 8,
        max_edge_repeats: int = 1,
        max_visits_per_agent: int | None = None,
        severity: Severity = Severity.BLOCK,
    ) -> None:
        super().__init__(severity=severity)
        if max_hops < 0 or max_edge_repeats < 1:
            raise ValueError("max_hops must be >= 0 and max_edge_repeats >= 1")
        if max_visits_per_agent is not None and max_visits_per_agent < 1:
            raise ValueError("max_visits_per_agent must be >= 1")
        self.max_hops = max_hops
        self.max_edge_repeats = max_edge_repeats
        self.max_visits_per_agent = max_visits_per_agent

    def check(self, trajectory: Trajectory, stage: Stage) -> AssertionResult:
        sequence = trajectory.agent_sequence()
        handoffs = trajectory.handoffs()
        hops = len(handoffs)
        visits = Counter(sequence)
        edges = Counter(handoffs)
        cycle = find_cycle(handoffs)
        metrics = {
            "hops": float(hops),
            "max_visits": float(max(visits.values(), default=0)),
            "max_edge_repeats": float(max(edges.values(), default=0)),
            "cyclic": float(cycle is not None),
        }

        problems: list[str] = []
        if hops > self.max_hops:
            problems.append(f"{hops} hops exceeds max_hops={self.max_hops}")
        repeated = sorted(e for e, n in edges.items() if n > self.max_edge_repeats)
        if repeated:
            detail = ", ".join(f"{a} -> {b} x{edges[(a, b)]}" for a, b in repeated)
            problems.append(
                f"handoff repeats exceed max_edge_repeats={self.max_edge_repeats}: {detail}"
            )
        if self.max_visits_per_agent is not None:
            crowded = sorted(a for a, n in visits.items() if n > self.max_visits_per_agent)
            if crowded:
                detail = ", ".join(f"{agent} x{visits[agent]}" for agent in crowded)
                problems.append(
                    f"agent visits exceed max_visits_per_agent={self.max_visits_per_agent}: "
                    f"{detail}"
                )
        if problems:
            evidence = ["route: " + " -> ".join(sequence)]
            if cycle:
                evidence.append("cycle: " + " -> ".join(cycle))
            return self.fail(stage, "; ".join(problems), evidence=evidence, **metrics)  # type: ignore[arg-type]
        return self.ok(stage, f"{hops} hop(s), no runaway routing", **metrics)


class NoContextBloat(BaseInvariant):
    """Bound input token growth across sequential handoffs.

    Looks at the input tokens of agent and model steps in order (rolled up
    parent usage is deduplicated, see ``Trajectory.token_attribution``).
    Fails when the peak is more than ``max_growth_factor`` times the baseline,
    or when cumulative input tokens exceed ``max_cumulative_input_tokens``.

    The baseline is the first step, but never less than
    ``min_baseline_tokens``. Without that floor a 50 token intent router
    followed by a normal 300 token prompt would read as 6x growth.
    """

    name: ClassVar[str] = "no_context_bloat"
    stages: ClassVar[frozenset[Stage]] = _RUNTIME_STAGES

    def __init__(
        self,
        *,
        max_growth_factor: float = 4.0,
        min_baseline_tokens: int = 500,
        max_cumulative_input_tokens: int | None = None,
        severity: Severity = Severity.BLOCK,
    ) -> None:
        super().__init__(severity=severity)
        if max_growth_factor < 1.0:
            raise ValueError("max_growth_factor must be >= 1.0")
        if min_baseline_tokens < 1:
            raise ValueError("min_baseline_tokens must be >= 1")
        self.max_growth_factor = max_growth_factor
        self.min_baseline_tokens = min_baseline_tokens
        self.max_cumulative_input_tokens = max_cumulative_input_tokens

    def check(self, trajectory: Trajectory, stage: Stage) -> AssertionResult:
        steps = sorted(
            (
                (span, input_tokens)
                for span, input_tokens, _ in trajectory.token_attribution()
                if span.kind in (SpanKind.AGENT, SpanKind.LLM) and input_tokens > 0
            ),
            key=lambda row: row[0].start_ns,
        )
        series = [tokens for _, tokens in steps]
        cumulative = sum(series)
        baseline = max(series[0], self.min_baseline_tokens) if series else self.min_baseline_tokens
        growth = (max(series) / baseline) if series else 0.0
        metrics = {
            "cumulative_input_tokens": float(cumulative),
            "growth_factor": round(growth, 3),
            "baseline_tokens": float(baseline),
        }

        problems: list[str] = []
        if len(series) >= 2 and growth > self.max_growth_factor:
            problems.append(
                f"input tokens grew {growth:.1f}x over a {baseline} token baseline, "
                f"limit is {self.max_growth_factor:.1f}x"
            )
        limit = self.max_cumulative_input_tokens
        if limit is not None and cumulative > limit:
            problems.append(f"cumulative input tokens {cumulative} exceeds {limit}")
        if problems:
            trail = " -> ".join(f"{s.agent_name or s.name}:{n}" for s, n in steps)
            return self.fail(
                stage,
                "; ".join(problems),
                evidence=[f"input tokens per step: {trail}"],
                span_ids=[s.span_id for s, _ in steps],
                **metrics,
            )
        return self.ok(stage, "input token growth within bounds", **metrics)


_ERROR_KEYS = ("error", "errors", "exception", "traceback")
# Phrases that say the agent could not complete the work. Generic words such
# as "sorry", "error" or "failed" are deliberately absent: "Sorry for the
# billing error! Your refund is approved" acknowledges nothing.
_DEFAULT_ACK = (
    r"\b(?:(?:was|were|am|is|are)\s+unable\s+to|unable\s+to|not\s+able\s+to|"
    r"(?:could\s?not|couldn['’]t|can(?:no|['’])t)\s+(?:\w+\s+){0,2}?"
    r"(?:reach|retrieve|access|complete|process|verify|confirm|find|look\s?up|load|fetch|connect)|"
    r"(?:is|are|was|were)\s+(?:currently\s+|temporarily\s+)?unavailable|"
    r"timed\s?out|time\s?out|try\s+again|"
    r"(?:have|has|will|been)\s+(?:be\s+)?escalated|escalat(?:e|ing)\s+(?:this|your|the)|"
    r"technical\s+(?:issue|difficulty|difficulties|problem)s?)\b"
)


def tool_span_failed(span: Span) -> bool:
    """A tool span failed if its status is ERROR or its output is an error payload."""
    if span.status is SpanStatus.ERROR:
        return True
    output: Any = span.output
    if isinstance(output, Mapping):
        return any(output.get(key) for key in _ERROR_KEYS)
    return False


class NoSwallowedToolFailures(BaseInvariant):
    """Tool failures must be retried, handled or acknowledged.

    A failed tool call counts as handled when one of these holds:

    * a later call to the same tool succeeded (retry),
    * the span carries the attribute ``error.handled = True``,
    * the run itself ended in an error (the failure propagated),
    * the final output says the work could not be completed
      (``acknowledgement_pattern``).

    The last rule is a text heuristic. The default pattern only accepts
    inability phrases ("we were unable to", "could not retrieve", "try again",
    "is unavailable") and ignores apologies. Set ``allow_acknowledgement=False``
    to accept only the structural signals: retry, ``error.handled`` or a
    propagated error.

    Otherwise the model produced a confident answer on top of a failed tool
    call, which is reported as a swallowed failure.
    """

    name: ClassVar[str] = "no_swallowed_tool_failures"
    stages: ClassVar[frozenset[Stage]] = frozenset({Stage.OUTPUT})

    def __init__(
        self,
        *,
        acknowledgement_pattern: str = _DEFAULT_ACK,
        allow_acknowledgement: bool = True,
        severity: Severity = Severity.BLOCK,
    ) -> None:
        super().__init__(severity=severity)
        self.allow_acknowledgement = allow_acknowledgement
        self._ack = re.compile(acknowledgement_pattern, re.IGNORECASE)

    def check(self, trajectory: Trajectory, stage: Stage) -> AssertionResult:
        tools = trajectory.tool_spans()
        unhandled: list[Span] = []
        for index, span in enumerate(tools):
            if not tool_span_failed(span):
                continue
            if span.attributes.get("error.handled") is True:
                continue
            retried = any(
                later.tool_name == span.tool_name and not tool_span_failed(later)
                for later in tools[index + 1 :]
            )
            if not retried:
                unhandled.append(span)

        failures = float(sum(1 for s in tools if tool_span_failed(s)))
        if not unhandled:
            return self.ok(stage, "no unhandled tool failures", tool_failures=failures)
        if trajectory.error is not None:
            return self.ok(stage, "tool failure propagated as a run error", tool_failures=failures)
        output_text = flatten_text(trajectory.final_output)
        if self.allow_acknowledgement and output_text and self._ack.search(output_text):
            return self.ok(
                stage, "tool failure acknowledged in final output", tool_failures=failures
            )
        return self.fail(
            stage,
            f"{len(unhandled)} tool failure(s) swallowed without retry or acknowledgement",
            evidence=[
                f"{s.tool_name or s.name}: {s.error or 'error payload returned'}" for s in unhandled
            ],
            span_ids=[s.span_id for s in unhandled],
            tool_failures=failures,
            swallowed=float(len(unhandled)),
        )


class SLABounds(BaseInvariant):
    """Hard latency and token budgets.

    Evaluated at every span boundary and at the end. The check is synchronous
    and cooperative: it fires at the first boundary after the budget is blown,
    it does not preempt a call that is still running.
    """

    name: ClassVar[str] = "sla_bounds"
    stages: ClassVar[frozenset[Stage]] = _RUNTIME_STAGES

    def __init__(
        self,
        *,
        max_latency_ms: float | None = None,
        max_tokens: int | None = None,
        severity: Severity = Severity.BLOCK,
    ) -> None:
        super().__init__(severity=severity)
        if max_latency_ms is None and max_tokens is None:
            raise ValueError("set max_latency_ms, max_tokens or both")
        self.max_latency_ms = max_latency_ms
        self.max_tokens = max_tokens

    def check(self, trajectory: Trajectory, stage: Stage) -> AssertionResult:
        latency = trajectory.latency_ms
        tokens = trajectory.total_tokens
        problems: list[str] = []
        if self.max_latency_ms is not None and latency > self.max_latency_ms:
            problems.append(f"latency {latency:.0f} ms exceeds {self.max_latency_ms:.0f} ms")
        if self.max_tokens is not None and tokens > self.max_tokens:
            problems.append(f"{tokens} tokens exceeds {self.max_tokens}")
        metrics = {"latency_ms": round(latency, 3), "total_tokens": float(tokens)}
        if problems:
            return self.fail(stage, "; ".join(problems), **metrics)  # type: ignore[arg-type]
        return self.ok(stage, "within SLA bounds", **metrics)


# --------------------------------------------------------------------------- #
# Assert style factories
# --------------------------------------------------------------------------- #

def assert_valid_dag(**kwargs: Any) -> ValidSpanDAG:
    return ValidSpanDAG(**kwargs)


def assert_no_routing_loops(max_hops: int = 8, **kwargs: Any) -> NoRoutingLoops:
    return NoRoutingLoops(max_hops=max_hops, **kwargs)


def assert_no_context_bloat(**kwargs: Any) -> NoContextBloat:
    return NoContextBloat(**kwargs)


def assert_no_swallowed_tool_failures(**kwargs: Any) -> NoSwallowedToolFailures:
    return NoSwallowedToolFailures(**kwargs)


def assert_sla_bounds(
    max_latency_ms: float | None = None, max_tokens: int | None = None, **kwargs: Any
) -> SLABounds:
    return SLABounds(max_latency_ms=max_latency_ms, max_tokens=max_tokens, **kwargs)
