"""Real LangGraph StateGraph wrapped by EvaluationSession.

Run from the repository root after ``pip install -e ".[langgraph]"``::

    python examples/langgraph_agent/graph.py

Nodes read the active ``EvaluationSession`` via ``current_session()`` so the
LangGraph state stays plain JSON-serializable data. No API key is needed.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_eval import (
    EvaluationSession,
    InvariantRegistry,
    InvariantViolation,
    current_session,
    default_registry,
)

LEDGER: dict[str, dict[str, Any]] = {
    "TXN-1001": {
        "transaction_id": "TXN-1001",
        "merchant": "Northwind Books",
        "amount": 42.50,
        "currency": "USD",
        "card_last4": "1111",
    },
}


class DisputeState(TypedDict, total=False):
    message: str
    transaction_id: str
    fault: str | None
    intent: str
    record: dict[str, Any] | None
    answer: str
    loops: int


def triage_node(state: DisputeState) -> DisputeState:
    session = current_session()
    assert session is not None
    with session.agent("triage", input=state["message"]) as step:
        step.set_tokens(320, 45)
        intent = "status" if "status" in state["message"].lower() else "refund"
        step.set_output({"intent": intent})
    return {"intent": intent}


def ledger_lookup_node(state: DisputeState) -> DisputeState:
    session = current_session()
    assert session is not None
    tx_id = state["transaction_id"]
    fault = state.get("fault")
    with session.agent("ledger_lookup", input={"transaction_id": tx_id}) as step:
        step.set_tokens(410, 50)
        tool_input: dict[str, Any] = {"transaction_id": tx_id}
        if fault == "pan_in_tool_input":
            tool_input["pan"] = "4111111111111111"
        with session.tool("ledger.get_transaction", input=tool_input) as call:
            record = LEDGER[tx_id]
            call.set_output(record)
        step.set_output({"found": True})
    return {"record": record, "loops": state.get("loops", 0) + 1}


def _route_after_ledger(state: DisputeState) -> Literal["triage", "resolution"]:
    if state.get("fault") == "routing_loop" and state.get("loops", 0) < 5:
        return "triage"
    return "resolution"


def resolution_node(state: DisputeState) -> DisputeState:
    session = current_session()
    assert session is not None
    record = state["record"]
    assert record is not None
    with session.agent("resolution", input={"intent": state["intent"]}) as step:
        step.set_tokens(520, 95)
        amount = float(record["amount"])
        if state.get("fault") == "hallucinated_refund":
            amount *= 10
        answer = (
            f"Your dispute for {record['transaction_id']} at {record['merchant']} is approved. "
            f"A refund of ${amount:,.2f} has been issued to card ending {record['card_last4']}."
        )
        step.set_output(answer)
    return {"answer": answer}


def build_graph() -> Any:
    """Compile a 3-node LangGraph workflow with a conditional cycle edge."""
    builder: StateGraph[DisputeState] = StateGraph(DisputeState)
    builder.add_node("triage", triage_node)
    builder.add_node("ledger_lookup", ledger_lookup_node)
    builder.add_node("resolution", resolution_node)

    builder.add_edge(START, "triage")
    builder.add_edge("triage", "ledger_lookup")
    builder.add_conditional_edges(
        "ledger_lookup",
        _route_after_ledger,
        {"triage": "triage", "resolution": "resolution"},
    )
    builder.add_edge("resolution", END)
    return builder.compile()


def run_dispute(
    message: str,
    transaction_id: str = "TXN-1001",
    *,
    fault: str | None = None,
    invariants: InvariantRegistry | None = None,
    graph: Any | None = None,
) -> tuple[str, EvaluationSession]:
    """Run the compiled LangGraph workflow inside an enforcing EvaluationSession."""
    compiled = graph or build_graph()
    registry = invariants or default_registry(max_hops=6)
    session = EvaluationSession("langgraph_dispute", invariants=registry, model="demo-model")
    with session:
        session.set_input({"message": message, "transaction_id": transaction_id})
        result = compiled.invoke(
            {"message": message, "transaction_id": transaction_id, "fault": fault}
        )
        answer = str(result["answer"])
        session.set_output(answer)
    return answer, session


def main() -> int:
    compiled = build_graph()

    print("1. Clean refund:")
    answer, session = run_dispute("Please refund TXN-1001.", graph=compiled)
    assert session.report is not None
    print(session.report.summary())
    print(f"   released: {answer}\n")

    scenarios = [
        ("2. Input gate (SSN in prompt):", "Refund TXN-1001, my SSN is 123-45-6789.", None),
        ("3. Mid-run span gate (LangGraph routing loop):", "Refund TXN-1001.", "routing_loop"),
        ("4. Output gate (hallucinated refund):", "Refund TXN-1001.", "hallucinated_refund"),
    ]
    for title, msg, fault in scenarios:
        print(title)
        try:
            run_dispute(msg, fault=fault, graph=compiled)
        except InvariantViolation as exc:
            print(exc.report.summary())
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
