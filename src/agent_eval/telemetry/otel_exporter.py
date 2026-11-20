"""OpenTelemetry export using the GenAI semantic conventions.

A trajectory is replayed into real OTel spans with its original timestamps:

* one root span ``invoke_agent {trajectory name}`` carrying the verdict,
* one child span per recorded span, named ``invoke_agent {agent}``,
  ``execute_tool {tool}`` or ``chat {model}``,
* one ``gen_ai.evaluation.result`` event on the root per invariant.

Ids line up with the evaluation report. When the trajectory ids are valid hex
(the default generated ones are), the exported OTel trace id and span ids are
the same values, so ``agent_eval.span_ids`` on a failed result points straight
at spans in the backend. Every span also carries ``agent_eval.span_id`` and,
when an invariant failed on it, ``agent_eval.violations``.

Payloads (prompts, tool arguments, completions) are never exported. Only
names, token counts, timings, verdicts and masked evidence leave the process.

The GenAI conventions are still marked as in development upstream, so the
attribute names are kept in one place below.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.id_generator import IdGenerator, RandomIdGenerator
from opentelemetry.trace import SpanKind as OtelSpanKind
from opentelemetry.trace import Status, StatusCode

from agent_eval._version import __version__
from agent_eval.schemas import EvalReport, Span, SpanKind, SpanStatus, Trajectory

# ---- GenAI semantic convention names ----------------------------------------
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
ERROR_TYPE = "error.type"

GEN_AI_EVALUATION_EVENT = "gen_ai.evaluation.result"
GEN_AI_EVALUATION_NAME = "gen_ai.evaluation.name"
GEN_AI_EVALUATION_SCORE_VALUE = "gen_ai.evaluation.score.value"
GEN_AI_EVALUATION_SCORE_LABEL = "gen_ai.evaluation.score.label"
GEN_AI_EVALUATION_EXPLANATION = "gen_ai.evaluation.explanation"

# ---- Project specific names (no convention exists for these) ----------------
EVAL_VERDICT = "agent_eval.verdict"
EVAL_BLOCKED_AT = "agent_eval.blocked_at"
EVAL_VIOLATIONS = "agent_eval.violations"
EVAL_VIOLATION_COUNT = "agent_eval.violation_count"
EVAL_SEVERITY = "agent_eval.severity"
EVAL_STAGE = "agent_eval.stage"
EVAL_EVIDENCE = "agent_eval.evidence"
EVAL_SPAN_IDS = "agent_eval.span_ids"
EVAL_SPAN_ID = "agent_eval.span_id"
EVAL_TRACE_ID = "agent_eval.trace_id"
TRAJECTORY_PREFIX = "agent_eval.trajectory."
METRIC_PREFIX = "agent_eval.metric."

_HEX_TRACE_ID = re.compile(r"^[0-9a-f]{32}$")
_HEX_SPAN_ID = re.compile(r"^[0-9a-f]{16}$")


class _ReplayIdGenerator(IdGenerator):
    """Hands out preset ids once, then falls back to random ones."""

    def __init__(self) -> None:
        self._random = RandomIdGenerator()
        self.next_trace_id: int | None = None
        self.next_span_id: int | None = None

    def generate_trace_id(self) -> int:
        value, self.next_trace_id = self.next_trace_id, None
        return value or self._random.generate_trace_id()

    def generate_span_id(self) -> int:
        value, self.next_span_id = self.next_span_id, None
        return value or self._random.generate_span_id()


class JsonFileSpanExporter(SpanExporter):
    """Writes one JSON object per span to a JSON Lines file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            with self._lock, self._path.open("a", encoding="utf-8") as handle:
                for span in spans:
                    record = json.loads(span.to_json(indent=None))
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError:
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def make_otlp_exporter(
    endpoint: str | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> SpanExporter:
    """Standard OTLP over HTTP (protobuf) trace exporter.

    With no arguments the exporter reads the usual environment variables:
    ``OTEL_EXPORTER_OTLP_ENDPOINT``, ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` and
    ``OTEL_EXPORTER_OTLP_HEADERS``. An explicit ``endpoint`` must be the full
    traces URL, for example ``http://localhost:4318/v1/traces``.
    """
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError as exc:
        raise ImportError(
            "OTLP export requires the 'otlp' extra: pip install 'agent-invariant-eval[otlp]'"
        ) from exc

    return OTLPSpanExporter(
        endpoint=endpoint,
        headers=dict(headers) if headers else None,
        timeout=timeout,
    )


def span_attributes(span: Span) -> dict[str, str | int | float | bool]:
    """GenAI attributes for one recorded span."""
    attributes: dict[str, str | int | float | bool] = dict(span.attributes)
    attributes[GEN_AI_OPERATION_NAME] = span.kind.value
    attributes[EVAL_SPAN_ID] = span.span_id
    if span.agent_name:
        attributes[GEN_AI_AGENT_NAME] = span.agent_name
    if span.tool_name:
        attributes[GEN_AI_TOOL_NAME] = span.tool_name
    if span.model:
        attributes[GEN_AI_REQUEST_MODEL] = span.model
    if span.input_tokens or span.output_tokens or span.kind is not SpanKind.TOOL:
        attributes[GEN_AI_USAGE_INPUT_TOKENS] = span.input_tokens
        attributes[GEN_AI_USAGE_OUTPUT_TOKENS] = span.output_tokens
    if span.status is SpanStatus.ERROR:
        attributes[ERROR_TYPE] = (span.error or "error").split(":", 1)[0]
    return attributes


class TelemetryEmitter:
    """Owns a private ``TracerProvider`` and replays trajectories into it.

    ``exporters`` are driven synchronously (good for files and tests).
    ``batch_exporters`` are driven by a background batch processor (use this
    for OTLP so network latency never sits on the evaluation path).
    The global tracer provider is left untouched.
    """

    def __init__(
        self,
        exporters: Iterable[SpanExporter] = (),
        *,
        batch_exporters: Iterable[SpanExporter] = (),
        service_name: str = "agent-invariant-eval",
        resource_attributes: Mapping[str, str] | None = None,
    ) -> None:
        resource = Resource.create({"service.name": service_name, **(resource_attributes or {})})
        self._ids = _ReplayIdGenerator()
        self._lock = threading.Lock()
        self._provider = TracerProvider(resource=resource, id_generator=self._ids)
        for exporter in exporters:
            self._provider.add_span_processor(SimpleSpanProcessor(exporter))
        for exporter in batch_exporters:
            self._provider.add_span_processor(BatchSpanProcessor(exporter))
        self._tracer = self._provider.get_tracer("agent_eval", __version__)

    def emit(self, trajectory: Trajectory, report: EvalReport | None = None) -> None:
        """Export one trajectory and, if given, its evaluation report."""
        with self._lock:  # the id generator is stateful
            self._emit(trajectory, report)

    def _emit(self, trajectory: Trajectory, report: EvalReport | None) -> None:
        starts = [s.start_ns for s in trajectory.spans if s.start_ns]
        ends = [s.end_ns for s in trajectory.spans if s.end_ns]
        start_ns = trajectory.start_ns or (min(starts) if starts else None)
        end_ns = trajectory.end_ns or (max(ends) if ends else None)

        root_attributes: dict[str, str | int | float | bool | list[str]] = {
            GEN_AI_OPERATION_NAME: SpanKind.AGENT.value,
            GEN_AI_AGENT_NAME: trajectory.name,
            EVAL_TRACE_ID: trajectory.trace_id,
            GEN_AI_USAGE_INPUT_TOKENS: trajectory.total_input_tokens,
            GEN_AI_USAGE_OUTPUT_TOKENS: trajectory.total_output_tokens,
        }
        if trajectory.model:
            root_attributes[GEN_AI_REQUEST_MODEL] = trajectory.model
        for key, value in trajectory.metrics().items():
            root_attributes[TRAJECTORY_PREFIX + key] = value

        if _HEX_TRACE_ID.match(trajectory.trace_id) and int(trajectory.trace_id, 16):
            self._ids.next_trace_id = int(trajectory.trace_id, 16)
        root = self._tracer.start_span(
            f"{SpanKind.AGENT.value} {trajectory.name}",
            context=Context(),  # always a new trace, never a child of ambient context
            kind=OtelSpanKind.INTERNAL,
            start_time=start_ns,
            attributes=root_attributes,
        )
        contexts: dict[str | None, Context] = {None: trace.set_span_in_context(root, Context())}

        try:
            ordered = trajectory.topological_spans()
        except ValueError:
            # Invalid DAG: still export, flat under the root.
            ordered = [s.model_copy(update={"parent_id": None}) for s in trajectory.ordered_spans()]

        failed_on: dict[str, list[str]] = {}
        if report is not None:
            for result in report.violations:
                for span_id in result.span_ids:
                    failed_on.setdefault(span_id, []).append(result.invariant)

        used_ids: set[str] = set()
        for span in ordered:
            parent = contexts.get(span.parent_id, contexts[None])
            attributes: dict[str, str | int | float | bool | list[str]] = dict(
                span_attributes(span)
            )
            if span.span_id in failed_on:
                attributes[EVAL_VIOLATIONS] = failed_on[span.span_id]
            reusable = (
                _HEX_SPAN_ID.match(span.span_id)
                and int(span.span_id, 16)
                and span.span_id not in used_ids
            )
            if reusable:
                self._ids.next_span_id = int(span.span_id, 16)
                used_ids.add(span.span_id)
            otel_span = self._tracer.start_span(
                span.name,
                context=parent,
                kind=OtelSpanKind.CLIENT if span.kind is SpanKind.LLM else OtelSpanKind.INTERNAL,
                start_time=span.start_ns or start_ns,
                attributes=attributes,
            )
            if span.status is SpanStatus.ERROR:
                otel_span.set_status(Status(StatusCode.ERROR, span.error or "error"))
            contexts[span.span_id] = trace.set_span_in_context(otel_span, Context())
            otel_span.end(end_time=span.end_ns or span.start_ns or end_ns)

        if report is not None:
            self._attach_report(root, report)
        if trajectory.error:
            root.set_attribute(ERROR_TYPE, trajectory.error.split(":", 1)[0])
        root.end(end_time=end_ns)

    @staticmethod
    def _attach_report(root: trace.Span, report: EvalReport) -> None:
        root.set_attribute(EVAL_VERDICT, report.verdict)
        root.set_attribute(EVAL_STAGE, report.stage.value)
        root.set_attribute(EVAL_VIOLATION_COUNT, len(report.violations))
        if report.violations:
            root.set_attribute(EVAL_VIOLATIONS, report.violated_names)
        if report.blocked_at is not None:
            root.set_attribute(EVAL_BLOCKED_AT, report.blocked_at.value)
        if not report.passed:
            root.set_status(Status(StatusCode.ERROR, "invariant violation"))

        for result in report.results:
            attributes: dict[str, str | int | float | bool | list[str]] = {
                GEN_AI_EVALUATION_NAME: result.invariant,
                GEN_AI_EVALUATION_SCORE_VALUE: 1.0 if result.passed else 0.0,
                GEN_AI_EVALUATION_SCORE_LABEL: "pass" if result.passed else "fail",
                EVAL_SEVERITY: result.severity.value,
                EVAL_STAGE: result.stage.value,
            }
            if result.message:
                attributes[GEN_AI_EVALUATION_EXPLANATION] = result.message
            if result.evidence:
                attributes[EVAL_EVIDENCE] = result.evidence
            if result.span_ids:
                attributes[EVAL_SPAN_IDS] = result.span_ids
            for key, value in result.metrics.items():
                attributes[METRIC_PREFIX + key] = value
            root.add_event(GEN_AI_EVALUATION_EVENT, attributes=attributes)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._provider.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._provider.shutdown()

    def __enter__(self) -> TelemetryEmitter:
        return self

    def __exit__(self, *_: object) -> None:
        self.force_flush()
        self.shutdown()
