"""Integration tests for the LangGraph example wrapped by EvaluationSession."""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from graph import build_graph, run_dispute  # noqa: E402

from agent_eval import InvariantViolation, SpanKind, Stage  # noqa: E402


def test_langgraph_clean_run_passes_and_records_dag():
    graph = build_graph()
    answer, session = run_dispute("Please refund TXN-1001.", graph=graph)
    assert "$42.50" in answer
    assert session.report is not None and session.report.passed is True
    assert session.trajectory.agent_sequence() == ["triage", "ledger_lookup", "resolution"]
    assert session.trajectory.dag_errors() == []
    tools = [s for s in session.trajectory.spans if s.kind is SpanKind.TOOL]
    assert len(tools) == 1
    assert tools[0].tool_name == "ledger.get_transaction"
    assert tools[0].agent_name == "ledger_lookup"


def test_langgraph_input_gate_blocks_before_any_node_runs():
    graph = build_graph()
    with pytest.raises(InvariantViolation) as info:
        run_dispute("Refund TXN-1001, my SSN is 123-45-6789.", graph=graph)
    assert info.value.report.blocked_at is Stage.INPUT
    assert info.value.report.violated_names == ["no_pii"]
    assert info.value.report.metrics["span_count"] == 0


def test_langgraph_pre_execution_tool_gate_blocks_pan_in_tool_input():
    graph = build_graph()
    with pytest.raises(InvariantViolation) as info:
        run_dispute("Refund TXN-1001.", fault="pan_in_tool_input", graph=graph)
    assert info.value.report.blocked_at is Stage.SPAN
    assert info.value.report.violated_names == ["pci_compliant"]


def test_langgraph_mid_run_gate_cuts_conditional_routing_loop():
    graph = build_graph()
    with pytest.raises(InvariantViolation) as info:
        run_dispute("Refund TXN-1001.", fault="routing_loop", graph=graph)
    assert info.value.report.blocked_at is Stage.SPAN
    assert info.value.report.violated_names == ["no_routing_loops"]


def test_langgraph_output_gate_blocks_hallucinated_refund():
    graph = build_graph()
    with pytest.raises(InvariantViolation) as info:
        run_dispute("Refund TXN-1001.", fault="hallucinated_refund", graph=graph)
    assert info.value.report.blocked_at is Stage.OUTPUT
    assert info.value.report.violated_names == ["numeric_grounding"]
