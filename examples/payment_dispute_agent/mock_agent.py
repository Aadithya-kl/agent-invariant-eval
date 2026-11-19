"""Simulated payment dispute workflow: Triage -> Ledger Lookup -> Resolution.

No model is called. Each "agent" is plain Python that records spans on an
``EvaluationSession`` the same way a real LangGraph node, CrewAI task or raw
API loop would. Model tiers differ in simulated latency, price, and in which
failure modes they are susceptible to.

Failure modes are injected from the dataset (``metadata.fault`` and
``metadata.fault_tiers``) so that runs are fully reproducible:

* ``hallucinated_refund``     resolution states an amount the ledger never returned
* ``routing_loop``            triage and ledger lookup bounce the case back and forth
* ``swallowed_tool_failure``  the ledger call times out; weak tiers answer anyway
* ``pii_echo``                resolution leaks a customer email into the answer
* ``context_bloat``           each handoff forwards the full, growing transcript
* ``slow_ledger``             the ledger backend is degraded and blows the latency SLA

All model names and prices are fictional and only illustrate the mechanics.
"""

from __future__ import annotations

import time
from typing import Any

from agent_eval import EvaluationSession, InvariantRegistry, SimClock, default_registry
from agent_eval.benchmark import EvalCase, ModelConfig

MAX_LATENCY_MS = 6_000
MAX_TOKENS = 20_000

# Fictional tiers. Prices are USD per million tokens.
MODELS: list[ModelConfig] = [
    ModelConfig(
        name="frontier-xl",
        input_cost_per_mtok=15.0,
        output_cost_per_mtok=75.0,
        params={"tier": "frontier", "llm_ms": 1100},
    ),
    ModelConfig(
        name="balanced-m",
        input_cost_per_mtok=3.0,
        output_cost_per_mtok=15.0,
        params={"tier": "balanced", "llm_ms": 450},
    ),
    ModelConfig(
        name="balanced-m-legacy",
        input_cost_per_mtok=4.0,
        output_cost_per_mtok=20.0,
        params={"tier": "balanced", "llm_ms": 700},
    ),
    ModelConfig(
        name="distilled-s",
        input_cost_per_mtok=0.25,
        output_cost_per_mtok=1.25,
        params={"tier": "distilled", "llm_ms": 160},
    ),
]

# The system of record. Tool payloads carry no PII and only the last four
# digits of the card, so nominal runs stay clean.
LEDGER: dict[str, dict[str, Any]] = {
    "TXN-1001": {
        "transaction_id": "TXN-1001",
        "merchant": "Northwind Books",
        "amount": 42.50,
        "currency": "USD",
        "status": "refund_eligible",
        "card_last4": "1111",
    },
    "TXN-1002": {
        "transaction_id": "TXN-1002",
        "merchant": "Contoso Travel",
        "amount": 1280.00,
        "currency": "USD",
        "status": "under_review",
        "card_last4": "4242",
    },
    "TXN-1003": {
        "transaction_id": "TXN-1003",
        "merchant": "Fabrikam Electronics",
        "amount": 219.99,
        "currency": "USD",
        "status": "refund_eligible",
        "card_last4": "0005",
    },
}

TOOL_MS = 120
SLOW_TOOL_MS = 9_000


def build_registry() -> InvariantRegistry:
    """Default financial invariants plus the SLA budget for this workflow."""
    return default_registry(max_latency_ms=MAX_LATENCY_MS, max_tokens=MAX_TOKENS, max_hops=6)


def _work(session: EvaluationSession, ms: float) -> None:
    """Spend ``ms`` of time: instantly on a simulated clock, for real otherwise."""
    if isinstance(session.clock, SimClock):
        session.clock.advance(ms)
    else:
        time.sleep(ms / 1000)


def _active_fault(case: EvalCase, tier: str) -> str | None:
    fault = case.metadata.get("fault")
    tiers = case.metadata.get("fault_tiers", [])
    if fault and ("*" in tiers or tier in tiers):
        return str(fault)
    return None


def _ledger_lookup(session: EvaluationSession, transaction_id: str, *, fail: bool = False,
                   slow: bool = False) -> dict[str, Any]:
    with session.tool("ledger.get_transaction", input={"transaction_id": transaction_id}) as call:
        _work(session, SLOW_TOOL_MS if slow else TOOL_MS)
        if fail:
            raise TimeoutError("ledger upstream timed out")
        record = LEDGER[transaction_id]
        call.set_output(record)
        return record


def run_dispute(case: EvalCase, model: ModelConfig, session: EvaluationSession) -> str:
    """Run one dispute through the three agent workflow and return the answer."""
    tier = str(model.params.get("tier", "frontier"))
    llm_ms = float(model.params.get("llm_ms", 500))
    fault = _active_fault(case, tier)
    message = str(case.input["message"])
    transaction_id = str(case.input["transaction_id"])
    bloat = fault == "context_bloat"

    # ---- Triage -------------------------------------------------------------
    with session.agent("triage", input=message) as step:
        _work(session, llm_ms)
        step.set_tokens(400 if bloat else 350, 60)
        intent = "status" if "status" in message.lower() else "refund"
        step.set_output({"intent": intent, "route": "ledger_lookup"})

    # ---- Ledger lookup ------------------------------------------------------
    record: dict[str, Any] | None = None
    with session.agent("ledger_lookup", input={"transaction_id": transaction_id}) as step:
        _work(session, llm_ms)
        step.set_tokens(2400 if bloat else 500, 60)
        if case.metadata.get("fault") == "swallowed_tool_failure":
            # The ledger flakes on the first call for every tier.
            try:
                _ledger_lookup(session, transaction_id, fail=True)
            except TimeoutError:
                if fault == "swallowed_tool_failure":
                    record = None  # weak tier: carries on with nothing
                else:
                    record = _ledger_lookup(session, transaction_id)  # strong tier: retries
        else:
            record = _ledger_lookup(session, transaction_id, slow=fault == "slow_ledger")
        step.set_output({"found": record is not None})

    # ---- Routing loop: triage and ledger lookup keep handing the case back ---
    if fault == "routing_loop":
        for _ in range(4):
            with session.agent("triage", input="re-triage requested") as step:
                _work(session, llm_ms)
                step.set_tokens(350, 40)
                step.set_output({"route": "ledger_lookup"})
            with session.agent("ledger_lookup", input={"transaction_id": transaction_id}) as step:
                _work(session, llm_ms)
                step.set_tokens(500, 40)
                step.set_output({"route": "triage"})

    # ---- Resolution ---------------------------------------------------------
    with session.agent("resolution", input={"intent": intent, "found": record is not None}) as step:
        _work(session, llm_ms)
        step.set_tokens(9600 if bloat else 650, 120)
        if record is None:
            answer = (
                f"Good news! Your dispute for {transaction_id} has been approved "
                "and your refund is on its way."
            )
        elif intent == "status":
            answer = (
                f"Your dispute for {transaction_id} at {record['merchant']} is currently "
                f"{record['status'].replace('_', ' ')}. No action is needed from you."
            )
        else:
            amount = float(record["amount"])
            if fault == "hallucinated_refund":
                amount = amount * 10  # the model "remembers" the wrong figure
            answer = (
                f"Your dispute for {transaction_id} at {record['merchant']} is approved. "
                f"A refund of ${amount:,.2f} has been issued to the card ending in "
                f"{record['card_last4']}."
            )
        if fault == "pii_echo":
            answer += " A confirmation was sent to jane.doe@example.com."
        step.set_output(answer)

    return answer
