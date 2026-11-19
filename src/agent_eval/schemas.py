"""Core data model: spans, trajectories, assertion results and evaluation reports.

A trajectory is a set of spans linked by ``parent_id``. The parent links must
form a directed acyclic graph (a forest). Agent to agent routing is derived
from the start order of agent spans, and that routing graph is allowed to
contain cycles, which is exactly what the topology invariants look for.
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

AttributeValue = str | int | float | bool


class SpanKind(StrEnum):
    """Span kinds. Values are the OpenTelemetry GenAI operation names."""

    AGENT = "invoke_agent"
    TOOL = "execute_tool"
    LLM = "chat"


class SpanStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


class Stage(StrEnum):
    """Points in an execution where invariants are evaluated."""

    INPUT = "input"    # before the agent runs
    SPAN = "span"      # at every span boundary (tool call, handoff, model call)
    OUTPUT = "output"  # on the final output, before it is released


class Severity(StrEnum):
    BLOCK = "block"  # failure blocks the execution
    WARN = "warn"    # failure is reported but does not block


def new_span_id() -> str:
    return uuid.uuid4().hex[:16]


def new_trace_id() -> str:
    return uuid.uuid4().hex


class Span(BaseModel):
    """One unit of work: an agent invocation, a tool execution or a model call."""

    model_config = ConfigDict(extra="forbid")

    span_id: str = Field(default_factory=new_span_id)
    parent_id: str | None = None
    kind: SpanKind
    name: str
    agent_name: str | None = None
    tool_name: str | None = None
    model: str | None = None
    input: Any = None
    output: Any = None
    # Tokens used by this span. A parent that repeats (rolls up) the usage of
    # its children is deduplicated by Trajectory.token_attribution().
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    start_ns: int = Field(default=0, ge=0)
    end_ns: int | None = Field(default=None, ge=0)
    status: SpanStatus = SpanStatus.OK
    error: str | None = None
    attributes: dict[str, AttributeValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_times(self) -> Span:
        if self.end_ns is not None and self.end_ns < self.start_ns:
            raise ValueError("span end_ns is before start_ns")
        return self

    @property
    def latency_ms(self) -> float:
        if self.end_ns is None:
            return 0.0
        return (self.end_ns - self.start_ns) / 1e6

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Trajectory(BaseModel):
    """A full (or in progress) agent execution trace."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(default_factory=new_trace_id)
    name: str = "agent"
    model: str | None = None
    input: Any = None
    final_output: Any = None
    spans: list[Span] = Field(default_factory=list)
    start_ns: int = Field(default=0, ge=0)
    end_ns: int | None = Field(default=None, ge=0)
    # Set when the wrapped agent raised an exception that was not handled.
    error: str | None = None

    # ---- ordering and selection -------------------------------------------------

    def ordered_spans(self) -> list[Span]:
        """Spans by start time. Ties keep insertion order."""
        return sorted(self.spans, key=lambda s: s.start_ns)

    def agent_spans(self) -> list[Span]:
        return [s for s in self.ordered_spans() if s.kind is SpanKind.AGENT]

    def tool_spans(self) -> list[Span]:
        return [s for s in self.ordered_spans() if s.kind is SpanKind.TOOL]

    def agent_sequence(self) -> list[str]:
        """Agent visit order with consecutive repeats collapsed.

        ``[triage, triage, ledger, triage]`` becomes ``[triage, ledger, triage]``.
        """
        sequence: list[str] = []
        for span in self.agent_spans():
            agent = span.agent_name or span.name
            if not sequence or sequence[-1] != agent:
                sequence.append(agent)
        return sequence

    def handoffs(self) -> list[tuple[str, str]]:
        """Directed agent to agent edges in the order they were taken."""
        sequence = self.agent_sequence()
        return list(zip(sequence, sequence[1:], strict=False))

    # ---- aggregates -------------------------------------------------------------

    def token_attribution(self) -> list[tuple[Span, int, int]]:
        """Token usage per span as ``(span, input_tokens, output_tokens)``, deduplicated.

        Frameworks often report usage on a model span and again, rolled up, on
        the enclosing agent span. Summing both doubles budgets and costs, so
        usage is attributed like this:

        * a span with no token reporting descendants keeps its own usage,
        * a span whose input and output usage are both at least the sum of its
          descendants is read as a roll up: the descendants keep theirs and
          the span keeps only the remainder (zero for an exact roll up),
        * a span that is smaller than its descendants on either side cannot be
          a roll up, so it is counted in full on top of them.

        If the span graph is not a valid DAG every span is counted in full,
        which can only over count (fail closed for budgets).
        """
        by_id = {s.span_id: s for s in self.spans}
        full = [(s, s.input_tokens, s.output_tokens) for s in self.spans if s.total_tokens]
        if len(by_id) != len(self.spans):
            return full
        children: dict[str, list[Span]] = {}
        roots: list[Span] = []
        for span in self.spans:
            if span.parent_id is not None and span.parent_id in by_id:
                children.setdefault(span.parent_id, []).append(span)
            else:
                roots.append(span)

        rows: list[tuple[Span, int, int]] = []
        visited: set[str] = set()

        def visit(span: Span) -> tuple[int, int]:
            visited.add(span.span_id)
            below_in = below_out = 0
            for child in children.get(span.span_id, ()):
                child_in, child_out = visit(child)
                below_in += child_in
                below_out += child_out
            own_in, own_out = span.input_tokens, span.output_tokens
            if (below_in or below_out) and own_in >= below_in and own_out >= below_out:
                own_in -= below_in
                own_out -= below_out
            if own_in or own_out:
                rows.append((span, own_in, own_out))
            return below_in + own_in, below_out + own_out

        for root in roots:
            visit(root)
        if len(visited) != len(self.spans):  # a parent cycle left spans unreachable
            return full
        return rows

    @property
    def total_input_tokens(self) -> int:
        return sum(row[1] for row in self.token_attribution())

    @property
    def total_output_tokens(self) -> int:
        return sum(row[2] for row in self.token_attribution())

    @property
    def total_tokens(self) -> int:
        return sum(row[1] + row[2] for row in self.token_attribution())

    @property
    def latency_ms(self) -> float:
        starts = [s.start_ns for s in self.spans]
        ends = [s.end_ns if s.end_ns is not None else s.start_ns for s in self.spans]
        start = self.start_ns or (min(starts) if starts else 0)
        end = self.end_ns if self.end_ns is not None else (max(ends) if ends else start)
        return max(0.0, (end - start) / 1e6)

    def metrics(self) -> dict[str, float]:
        tools = self.tool_spans()
        usage = self.token_attribution()
        input_tokens = sum(row[1] for row in usage)
        output_tokens = sum(row[2] for row in usage)
        return {
            "latency_ms": round(self.latency_ms, 3),
            "input_tokens": float(input_tokens),
            "output_tokens": float(output_tokens),
            "total_tokens": float(input_tokens + output_tokens),
            "span_count": float(len(self.spans)),
            "hops": float(len(self.handoffs())),
            "tool_calls": float(len(tools)),
            "tool_errors": float(sum(1 for s in tools if s.status is SpanStatus.ERROR)),
        }

    # ---- DAG structure ----------------------------------------------------------

    def dag_errors(self) -> list[str]:
        """Structural problems in the span parent graph. Empty means a valid DAG."""
        errors: list[str] = []
        seen: set[str] = set()
        for span in self.spans:
            if span.span_id in seen:
                errors.append(f"duplicate span_id {span.span_id}")
            seen.add(span.span_id)
        parents = {s.span_id: s.parent_id for s in self.spans}
        for span_id, parent_id in parents.items():
            if parent_id is not None and parent_id not in parents:
                errors.append(f"span {span_id} has unknown parent {parent_id}")
        for span_id in parents:
            path = {span_id}
            cursor = parents.get(span_id)
            while cursor is not None and cursor in parents:
                if cursor in path:
                    errors.append(f"parent cycle through span {span_id}")
                    break
                path.add(cursor)
                cursor = parents.get(cursor)
        return errors

    def topological_spans(self) -> list[Span]:
        """Spans with every parent before its children. Raises on an invalid DAG."""
        problems = self.dag_errors()
        if problems:
            raise ValueError("invalid span DAG: " + "; ".join(problems))
        by_id = {s.span_id: s for s in self.spans}
        emitted: set[str] = set()
        result: list[Span] = []

        def visit(span: Span) -> None:
            if span.span_id in emitted:
                return
            if span.parent_id is not None:
                visit(by_id[span.parent_id])
            emitted.add(span.span_id)
            result.append(span)

        for span in self.ordered_spans():
            visit(span)
        return result


class AssertionResult(BaseModel):
    """Outcome of one invariant on one trajectory."""

    model_config = ConfigDict(extra="forbid")

    invariant: str
    passed: bool
    severity: Severity = Severity.BLOCK
    stage: Stage = Stage.OUTPUT
    message: str = ""
    # Evidence is exported to telemetry. Invariants must mask sensitive values.
    evidence: list[str] = Field(default_factory=list)
    span_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class EvalReport(BaseModel):
    """Verdict for one execution."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    name: str = "agent"
    model: str | None = None
    stage: Stage = Stage.OUTPUT
    passed: bool
    blocked_at: Stage | None = None
    results: list[AssertionResult] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)

    @property
    def violations(self) -> list[AssertionResult]:
        return [r for r in self.results if not r.passed]

    @property
    def violated_names(self) -> list[str]:
        return [r.invariant for r in self.violations]

    @property
    def verdict(self) -> str:
        return "pass" if self.passed else "block"

    def summary(self) -> str:
        head = f"[{self.name}] verdict={self.verdict}"
        if self.blocked_at is not None:
            head += f" blocked_at={self.blocked_at.value}"
        lines = [head]
        for result in self.results:
            mark = "PASS" if result.passed else "FAIL"
            line = f"  {mark} {result.invariant}"
            if result.message:
                line += f": {result.message}"
            lines.append(line)
            for item in result.evidence:
                lines.append(f"       evidence: {item}")
        return "\n".join(lines)
