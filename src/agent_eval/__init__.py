"""agent-invariant-eval: deterministic, fail-closed evaluation for multi-agent workflows."""

from agent_eval._version import __version__
from agent_eval.decorators import (
    EvaluationSession,
    SimClock,
    SpanScope,
    current_session,
    evaluate_agent,
)
from agent_eval.invariants import (
    BaseInvariant,
    InvariantRegistry,
    InvariantViolation,
    assert_no_context_bloat,
    assert_no_pii,
    assert_no_routing_loops,
    assert_no_swallowed_tool_failures,
    assert_numeric_grounding,
    assert_pci_compliant,
    assert_sla_bounds,
    assert_valid_dag,
    default_registry,
)
from agent_eval.schemas import (
    AssertionResult,
    EvalReport,
    Severity,
    Span,
    SpanKind,
    SpanStatus,
    Stage,
    Trajectory,
)

__all__ = [
    "AssertionResult",
    "BaseInvariant",
    "EvalReport",
    "EvaluationSession",
    "InvariantRegistry",
    "InvariantViolation",
    "Severity",
    "SimClock",
    "Span",
    "SpanKind",
    "SpanScope",
    "SpanStatus",
    "Stage",
    "Trajectory",
    "__version__",
    "assert_no_context_bloat",
    "assert_no_pii",
    "assert_no_routing_loops",
    "assert_no_swallowed_tool_failures",
    "assert_numeric_grounding",
    "assert_pci_compliant",
    "assert_sla_bounds",
    "assert_valid_dag",
    "current_session",
    "default_registry",
    "evaluate_agent",
]
