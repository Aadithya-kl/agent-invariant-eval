"""EvaluationSession, @evaluate_agent, telemetry export and the bundled example."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent_eval import (
    EvaluationSession,
    InvariantViolation,
    SimClock,
    Span,
    SpanKind,
    SpanStatus,
    Stage,
    assert_no_pii,
    assert_no_routing_loops,
    assert_numeric_grounding,
    assert_sla_bounds,
    current_session,
    default_registry,
    evaluate_agent,
)
from agent_eval.benchmark import FrontierRunner, load_dataset
from agent_eval.telemetry import JsonFileSpanExporter, TelemetryEmitter

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "payment_dispute_agent"


# ---- session --------------------------------------------------------------------

def test_clean_session_passes_and_builds_a_dag():
    with EvaluationSession("demo", invariants=default_registry(), clock=SimClock()) as session:
        session.set_input("refund TXN-1")
        with session.agent("triage") as step:
            session.clock.advance(100)
            step.set_tokens(100, 10)
            with session.tool("ledger", input={"id": "TXN-1"}) as call:
                session.clock.advance(50)
                call.set_output({"amount": 10.0})
        session.set_output("Refunded $10.00.")

    report = session.report
    assert report is not None and report.passed
    agent_span, tool_span = session.trajectory.spans
    assert tool_span.parent_id == agent_span.span_id
    assert tool_span.agent_name == "triage"
    assert report.metrics["latency_ms"] == 150
    assert report.metrics["total_tokens"] == 110


def test_input_gate_blocks_before_agent_code_runs():
    ran = []
    session = EvaluationSession(invariants=[assert_no_pii()])
    with pytest.raises(InvariantViolation) as info, session:
        session.set_input("my ssn is 123-45-6789")
        ran.append("agent")
    assert ran == []
    assert info.value.report.blocked_at is Stage.INPUT
    assert session.report is info.value.report


def test_span_gate_cuts_a_routing_loop_mid_run():
    session = EvaluationSession(invariants=[assert_no_routing_loops(max_hops=10)])
    visited = 0
    with pytest.raises(InvariantViolation) as info, session:
        for name in ["a", "b", "a", "b", "a", "b", "a", "b"]:
            with session.agent(name):
                visited += 1
    assert visited == 4  # blocked when the handoff a -> b happened a second time
    assert info.value.report.blocked_at is Stage.SPAN


def test_output_gate_blocks_hallucinated_amount():
    session = EvaluationSession(invariants=[assert_numeric_grounding()])
    with pytest.raises(InvariantViolation), session:
        with session.tool("ledger") as call:
            call.set_output({"amount": 10})
        session.set_output("Refunded $99.")
    assert session.report.blocked_at is Stage.OUTPUT


def test_sla_hard_fails_at_span_boundary():
    clock = SimClock()
    session = EvaluationSession(invariants=[assert_sla_bounds(max_latency_ms=1000)], clock=clock)
    with pytest.raises(InvariantViolation), session:
        with session.tool("slow"):
            clock.advance(1500)
        raise AssertionError("unreachable: the session must block first")
    assert session.report.violated_names == ["sla_bounds"]


def test_monitor_mode_reports_without_raising():
    with EvaluationSession(invariants=[assert_no_pii()], enforce=False) as session:
        session.set_input("mail jane.doe@example.com")
        session.set_output("ok")
    assert session.report.passed is False
    assert session.report.violated_names == ["no_pii"]


def test_tool_exception_marks_span_and_propagates():
    session = EvaluationSession(invariants=default_registry())
    with pytest.raises(TimeoutError), session, session.tool("ledger"):
        raise TimeoutError("upstream")
    span = session.trajectory.spans[0]
    assert span.status is SpanStatus.ERROR
    assert span.error == "TimeoutError: upstream"
    # The agent crashed, so the run fails closed.
    assert session.report.passed is False
    assert "agent_completed_without_error" in session.report.violated_names


def test_record_span_for_callback_style_integrations():
    with EvaluationSession(invariants=default_registry(), clock=SimClock()) as session:
        session.record_span(
            Span(kind=SpanKind.TOOL, name="execute_tool fx", tool_name="fx", output={"rate": 1.1})
        )
        session.set_output("done")
    assert session.report.passed
    assert session.trajectory.spans[0].end_ns is not None


# ---- decorator ------------------------------------------------------------------

def test_decorator_returns_output_and_stores_report():
    @evaluate_agent(invariants=default_registry())
    def agent(question: str) -> str:
        assert current_session() is not None
        return f"echo: {question}"

    assert agent("hello") == "echo: hello"
    assert agent.last_report.passed
    assert current_session() is None


def test_bare_decorator_uses_default_registry():
    @evaluate_agent
    def agent(question: str) -> str:
        return "fine"

    assert agent("hi") == "fine"


def test_decorator_injects_session_and_blocks_output():
    @evaluate_agent(invariants=default_registry())
    def agent(question: str, session: EvaluationSession) -> str:
        with session.tool("ledger") as call:
            call.set_output({"amount": 10})
        return "Refunded $1,000."

    with pytest.raises(InvariantViolation):
        agent("refund please")
    assert agent.last_report.violated_names == ["numeric_grounding"]


def test_decorator_blocks_pii_in_arguments():
    calls = []

    @evaluate_agent(invariants=default_registry())
    def agent(question: str, account: str = "") -> str:
        calls.append(question)
        return "ok"

    with pytest.raises(InvariantViolation):
        agent("hello", account="routing 011000015")
    assert calls == []


def test_decorator_supports_async_agents():
    @evaluate_agent(invariants=default_registry())
    async def agent(question: str) -> str:
        await asyncio.sleep(0)
        return "Contact jane.doe@example.com"

    with pytest.raises(InvariantViolation):
        asyncio.run(agent("hi"))
    assert agent.last_report.violated_names == ["no_pii"]


# ---- telemetry ------------------------------------------------------------------

def test_otel_spans_follow_genai_conventions():
    exporter = InMemorySpanExporter()
    emitter = TelemetryEmitter(exporters=[exporter])
    session = EvaluationSession(
        "dispute", invariants=default_registry(), emitter=emitter, clock=SimClock(), model="m-1"
    )
    with pytest.raises(InvariantViolation), session:
        session.set_input("refund")
        with session.agent("triage") as step:
            session.clock.advance(10)
            step.set_tokens(120, 30)
            with session.tool("ledger") as call:
                call.set_output({"amount": 10})
        session.set_output("Refunded $99 to jane.doe@example.com")

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {"invoke_agent dispute", "invoke_agent triage", "execute_tool ledger"}

    agent_span = spans["invoke_agent triage"]
    assert agent_span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert agent_span.attributes["gen_ai.agent.name"] == "triage"
    assert agent_span.attributes["gen_ai.request.model"] == "m-1"
    assert agent_span.attributes["gen_ai.usage.input_tokens"] == 120
    assert agent_span.attributes["gen_ai.usage.output_tokens"] == 30
    assert agent_span.end_time - agent_span.start_time == 10_000_000

    tool_span = spans["execute_tool ledger"]
    assert tool_span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool_span.attributes["gen_ai.tool.name"] == "ledger"
    assert tool_span.parent.span_id == agent_span.context.span_id

    root = spans["invoke_agent dispute"]
    assert agent_span.parent.span_id == root.context.span_id

    # OTel ids are the trajectory ids, so report span ids resolve in the backend.
    recorded_agent, recorded_tool = session.trajectory.spans
    assert format(root.context.trace_id, "032x") == session.trajectory.trace_id
    assert format(agent_span.context.span_id, "016x") == recorded_agent.span_id
    assert format(tool_span.context.span_id, "016x") == recorded_tool.span_id
    assert tool_span.attributes["agent_eval.span_id"] == recorded_tool.span_id
    assert root.attributes["agent_eval.trace_id"] == session.trajectory.trace_id
    assert root.attributes["agent_eval.verdict"] == "block"
    assert set(root.attributes["agent_eval.violations"]) == {"no_pii", "numeric_grounding"}
    events = {e.attributes["gen_ai.evaluation.name"]: e for e in root.events}
    assert all(e.name == "gen_ai.evaluation.result" for e in root.events)
    assert events["no_pii"].attributes["gen_ai.evaluation.score.label"] == "fail"
    assert events["pci_compliant"].attributes["gen_ai.evaluation.score.label"] == "pass"

    # No payloads and no raw PII may reach telemetry.
    dump = json.dumps([json.loads(s.to_json()) for s in spans.values()])
    assert "jane.doe" not in dump and "Refunded" not in dump


def test_json_file_exporter_writes_jsonl(tmp_path):
    path = tmp_path / "nested" / "spans.jsonl"
    emitter = TelemetryEmitter(exporters=[JsonFileSpanExporter(path)])
    with EvaluationSession("t", invariants=default_registry(), emitter=emitter) as session:
        with session.agent("a"):
            pass
        session.set_output("ok")
    emitter.shutdown()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["name"] for r in records] == ["invoke_agent a", "invoke_agent t"]
    assert records[0]["attributes"]["gen_ai.operation.name"] == "invoke_agent"
    assert records[1]["attributes"]["agent_eval.verdict"] == "pass"


# ---- bundled example ------------------------------------------------------------

def test_payment_dispute_example_end_to_end():
    from mock_agent import MODELS, build_registry, run_dispute

    cases = load_dataset(EXAMPLE / "test_suite.jsonl")
    assert len(cases) == 10
    runner = FrontierRunner(
        run_dispute, invariants=build_registry(), min_pass_rate=0.9, clock_factory=SimClock
    )
    report = runner.run(cases, MODELS)
    by_model = {s.model: s for s in report.summaries}

    assert by_model["frontier-xl"].passed == 10
    assert by_model["balanced-m"].passed == 9
    assert by_model["balanced-m-legacy"].passed == 9
    assert by_model["distilled-s"].passed == 5
    assert by_model["balanced-m-legacy"].pareto_optimal is False
    assert report.recommended == "balanced-m"

    failures = {
        (r.case_id, tuple(r.violations))
        for r in report.results
        if r.model == "distilled-s" and not r.passed
    }
    assert failures == {
        ("hallucinated-refund-amount", ("numeric_grounding",)),
        ("routing-loop", ("no_routing_loops",)),
        ("swallowed-ledger-timeout", ("no_swallowed_tool_failures",)),
        ("pii-echo-in-answer", ("no_pii",)),
        ("context-bloat-handoffs", ("no_context_bloat",)),
    }


# ---- Review fixes: concurrency, streaming, reports, ids, CLI --------------------

def test_parallel_async_branches_keep_their_own_parents():
    async def worker(session, name):
        with session.agent(name) as step:
            await asyncio.sleep(0)  # force interleaving
            with session.tool(f"{name}.tool") as call:
                await asyncio.sleep(0)
                call.set_output({"ok": True})
            await asyncio.sleep(0)
            step.set_tokens(10, 1)

    async def main():
        with EvaluationSession(invariants=default_registry()) as session:
            with session.agent("supervisor"):
                await asyncio.gather(*(worker(session, f"w{i}") for i in range(8)))
            session.set_output("done")
        return session

    session = asyncio.run(main())
    spans = {s.name: s for s in session.trajectory.spans}
    supervisor = spans["invoke_agent supervisor"]
    for index in range(8):
        agent_span = spans[f"invoke_agent w{index}"]
        tool_span = spans[f"execute_tool w{index}.tool"]
        assert agent_span.parent_id == supervisor.span_id
        assert tool_span.parent_id == agent_span.span_id
        assert tool_span.agent_name == f"w{index}"
    assert session.report.passed


def test_threads_record_safely_with_explicit_parents():
    from concurrent.futures import ThreadPoolExecutor

    with EvaluationSession(invariants=default_registry()) as session:
        with session.agent("supervisor") as parent:
            def work(index):
                with session.tool(f"t{index}", parent_id=parent.span.span_id) as call:
                    call.set_output({"i": index})

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(work, range(64)))
        session.set_output("done")
    assert len(session.trajectory.spans) == 65
    assert session.trajectory.dag_errors() == []
    assert session.report.passed


def test_generator_agent_is_buffered_and_blocked_before_any_chunk():
    released = []

    @evaluate_agent(invariants=default_registry())
    def stream(question: str):
        yield "Contact "
        yield "jane.doe@example.com"

    with pytest.raises(InvariantViolation):
        for chunk in stream("hi"):
            released.append(chunk)
    assert released == []
    assert stream.last_report.violated_names == ["no_pii"]


def test_clean_generator_streams_all_chunks():
    @evaluate_agent(invariants=default_registry())
    def stream(question: str, session: EvaluationSession):
        with session.tool("ledger") as call:
            call.set_output({"amount": 10})
        yield "Refunded "
        yield "$10."

    assert "".join(stream("hi")) == "Refunded $10."
    assert stream.last_report.passed


def test_passthrough_stream_yields_live_then_raises():
    released = []

    @evaluate_agent(invariants=default_registry(), stream="passthrough")
    def stream(question: str):
        yield "Contact "
        yield "jane.doe@example.com"
        yield " later"

    with pytest.raises(InvariantViolation):
        for chunk in stream("hi"):
            released.append(chunk)
    assert released == ["Contact ", "jane.doe@example.com", " later"]

    gen = stream("hi")
    assert [next(gen), next(gen)] == ["Contact ", "jane.doe@example.com"]
    gen.close()
    assert stream.last_report.violated_names == ["no_pii"]


def test_async_generator_agent_supported():
    @evaluate_agent(invariants=default_registry())
    async def stream(question: str):
        yield "all "
        await asyncio.sleep(0)
        yield "good"

    async def collect():
        return [chunk async for chunk in stream("hi")]

    assert asyncio.run(collect()) == ["all ", "good"]

    @evaluate_agent(invariants=default_registry())
    async def leaky(question: str):
        yield "ssn 123-45-6789"

    async def collect_leaky():
        return [chunk async for chunk in leaky("hi")]

    with pytest.raises(InvariantViolation):
        asyncio.run(collect_leaky())


def test_function_returning_a_generator_fails_closed():
    def inner():
        yield "x"

    @evaluate_agent(invariants=default_registry())
    def agent(question: str):
        return inner()

    with pytest.raises(TypeError, match="generator"):
        agent("hi")
    assert agent.last_report.passed is False


def test_with_report_is_race_free_under_concurrency():
    from concurrent.futures import ThreadPoolExecutor

    @evaluate_agent(invariants=default_registry(), enforce=False)
    def agent(index: int) -> str:
        return "mail jane.doe@example.com" if index % 2 else "ok"

    def call(index):
        result, report = agent.with_report(index)
        return index, result, report.passed

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(call, range(200)))
    assert all(passed == (index % 2 == 0) for index, _, passed in outcomes)


def test_async_with_report():
    @evaluate_agent(invariants=default_registry())
    async def agent(question: str) -> str:
        return "ok"

    result, report = asyncio.run(agent.with_report("hi"))
    assert result == "ok" and report.passed


def test_violating_span_is_tagged_in_telemetry():
    exporter = InMemorySpanExporter()
    emitter = TelemetryEmitter(exporters=[exporter])
    session = EvaluationSession(invariants=default_registry(), emitter=emitter, enforce=False)
    with session:
        with session.tool("crm") as call:
            call.set_output({"email": "jane.doe@example.com"})
        session.set_output("done")
    spans = {s.name: s for s in exporter.get_finished_spans()}
    tool_span = spans["execute_tool crm"]
    assert list(tool_span.attributes["agent_eval.violations"]) == ["no_pii"]
    event = next(e for e in spans["invoke_agent agent"].events
                 if e.attributes["gen_ai.evaluation.name"] == "no_pii")
    assert list(event.attributes["agent_eval.span_ids"]) == [
        format(tool_span.context.span_id, "016x")
    ]


def test_non_hex_span_ids_still_export():
    exporter = InMemorySpanExporter()
    emitter = TelemetryEmitter(exporters=[exporter])
    with EvaluationSession(invariants=default_registry(), emitter=emitter) as session:
        session.record_span(Span(span_id="custom-id", kind=SpanKind.TOOL, name="execute_tool x"))
        session.set_output("ok")
    span = next(s for s in exporter.get_finished_spans() if s.name == "execute_tool x")
    assert span.attributes["agent_eval.span_id"] == "custom-id"


# ---- CLI ------------------------------------------------------------------------

def test_cli_scan():
    from typer.testing import CliRunner

    from agent_eval.cli import app

    runner = CliRunner()
    clean = runner.invoke(app, ["scan", "refund of $42.50 for TXN-1001"])
    assert clean.exit_code == 0 and "clean" in clean.output

    dirty = runner.invoke(app, ["scan", "card 4111 1111 1111 1111, CVV 123"])
    assert dirty.exit_code == 1
    assert "************1111" in dirty.output and "4111 1111" not in dirty.output

    assert runner.invoke(app, ["scan"]).exit_code != 0


def test_cli_scan_file(tmp_path):
    from typer.testing import CliRunner

    from agent_eval.cli import app

    path = tmp_path / "note.txt"
    path.write_text("mail jane.doe@example.com")
    result = CliRunner().invoke(app, ["scan", "--file", str(path)])
    assert result.exit_code == 1 and "j***@example.com" in result.output


@pytest.mark.parametrize("models", ["mock_agent:MODELS", str(EXAMPLE / "models.json")])
def test_cli_benchmark_accepts_module_reference_and_json(models, tmp_path):
    from typer.testing import CliRunner

    from agent_eval.cli import app

    out = tmp_path / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "benchmark", str(EXAMPLE / "test_suite.jsonl"),
            "--agent", "mock_agent:run_dispute",
            "--registry", "mock_agent:build_registry",
            "--models", models,
            "--sim-clock",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "recommended: balanced-m" in result.output
    assert json.loads(out.read_text())["recommended"] == "balanced-m"


def test_cli_benchmark_rejects_bad_models_reference():
    from typer.testing import CliRunner

    from agent_eval.cli import app

    result = CliRunner().invoke(
        app,
        ["benchmark", str(EXAMPLE / "test_suite.jsonl"), "--agent", "mock_agent:run_dispute",
         "--models", "missing.json"],
    )
    assert result.exit_code != 0


def test_on_report_error_is_isolated_and_emitter_supports_context_manager(tmp_path):
    def broken_callback(_report):
        raise RuntimeError("callback boom")

    path = tmp_path / "cm_spans.jsonl"
    with TelemetryEmitter(exporters=[JsonFileSpanExporter(path)]) as emitter:
        with EvaluationSession(
            "cm", invariants=default_registry(), emitter=emitter, on_report=broken_callback
        ) as session:
            session.set_output("ok")
    assert session.report is not None and session.report.passed
    assert len(path.read_text().splitlines()) == 1
