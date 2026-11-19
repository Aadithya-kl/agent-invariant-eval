"""End to end demo: fail-closed blocking, frontier benchmark and OTel export.

Run from the repository root::

    python examples/payment_dispute_agent/run_eval.py

Spans are written to ``out/otel_spans.jsonl``. To also ship them over OTLP,
set ``OTEL_EXPORTER_OTLP_ENDPOINT`` (and ``OTEL_EXPORTER_OTLP_HEADERS`` for
vendor tokens) before running.
"""

from __future__ import annotations

import os
from pathlib import Path

from mock_agent import MODELS, build_registry, run_dispute

from agent_eval import EvaluationSession, InvariantViolation, SimClock, evaluate_agent
from agent_eval.benchmark import EvalCase, FrontierRunner, load_dataset
from agent_eval.telemetry import JsonFileSpanExporter, TelemetryEmitter, make_otlp_exporter

HERE = Path(__file__).resolve().parent
OUT = Path.cwd() / "out"


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    spans_path = OUT / "otel_spans.jsonl"
    spans_path.unlink(missing_ok=True)

    batch_exporters = []
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    ):
        batch_exporters.append(make_otlp_exporter())
    emitter = TelemetryEmitter(
        exporters=[JsonFileSpanExporter(spans_path)],
        batch_exporters=batch_exporters,
        service_name="payment-dispute-agent",
    )

    cases = {case.id: case for case in load_dataset(HERE / "test_suite.jsonl")}
    by_name = {model.name: model for model in MODELS}
    registry = build_registry()

    # ------------------------------------------------------------------ 1
    banner("1. Input gate: an SSN in the prompt is blocked before any agent runs")
    case = cases["adversarial-ssn-in-prompt"]
    session = EvaluationSession(
        case.id, invariants=registry, emitter=emitter, clock=SimClock(), model="frontier-xl"
    )
    try:
        with session:
            session.set_input(case.input)
            session.set_output(run_dispute(case, by_name["frontier-xl"], session))
    except InvariantViolation as violation:
        print(violation.report.summary())
        print(f"\nspans recorded before the block: {len(session.trajectory.spans)}")

    # ------------------------------------------------------------------ 2
    banner("2. Output gate: @evaluate_agent stops a hallucinated refund amount")
    hallucination_case = cases["hallucinated-refund-amount"]

    @evaluate_agent(
        name="payment_dispute",
        invariants=registry,
        emitter=emitter,
        clock=SimClock(),
        model="distilled-s",
    )
    def answer_dispute(request: dict, session: EvaluationSession) -> str:
        case = EvalCase(id="adhoc", input=request, metadata=hallucination_case.metadata)
        return run_dispute(case, by_name["distilled-s"], session)

    try:
        answer = answer_dispute(hallucination_case.input)
        print(f"released to caller: {answer}")
    except InvariantViolation as violation:
        print(violation.report.summary())
        print("\nthe answer was never returned to the caller")

    # ------------------------------------------------------------------ 3
    banner("3. Runtime gate: a routing loop is cut mid run at the span boundary")
    case = cases["routing-loop"]
    session = EvaluationSession(
        case.id, invariants=registry, emitter=emitter, clock=SimClock(), model="distilled-s"
    )
    try:
        with session:
            session.set_input(case.input)
            session.set_output(run_dispute(case, by_name["distilled-s"], session))
    except InvariantViolation as violation:
        print(violation.report.summary())

    # ------------------------------------------------------------------ 4
    banner("4. Cost to performance frontier across model configurations")
    runner = FrontierRunner(
        run_dispute,
        invariants=registry,
        min_pass_rate=0.9,
        clock_factory=SimClock,
        emitter=emitter,
    )
    report = runner.run(list(cases.values()), MODELS)
    print(report.to_markdown())
    print(f"\nrecommended configuration: {report.recommended}")

    print("\nfailed cases per model:")
    for result in report.results:
        if not result.passed:
            print(
                f"  {result.model:<18} {result.case_id:<28} "
                f"verdict={result.verdict} violations={result.violations}"
            )

    report_path = OUT / "frontier_report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ 5
    emitter.force_flush()
    emitter.shutdown()
    span_count = sum(1 for _ in spans_path.open(encoding="utf-8"))
    banner("5. OpenTelemetry export")
    print(f"{span_count} GenAI spans written to {spans_path}")
    print(f"frontier report written to {report_path}")
    if batch_exporters:
        print("spans also sent over OTLP")
    else:
        print("set OTEL_EXPORTER_OTLP_ENDPOINT to also send spans over OTLP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
