"""Runtime hooks: ``EvaluationSession`` and ``@evaluate_agent``.

The session is framework agnostic. It knows nothing about LangGraph, CrewAI or
any SDK. You tell it what happened (an agent ran, a tool was called, a model
was invoked) through small context managers, or by recording finished spans
from a framework callback, and it evaluates the registered invariants at three
points:

* ``Stage.INPUT``  when the input is set, before any agent work,
* ``Stage.SPAN``   every time a span closes (tool call, handoff, model call),
* ``Stage.OUTPUT`` on the final output, before it is released to the caller.

With ``enforce=True`` (the default) a blocking failure raises
``InvariantViolation`` at that point, so the output never leaves the session.

Concurrency: the stack of open spans lives in a ``contextvars.ContextVar``, so
branches started with ``asyncio.gather``, ``asyncio.TaskGroup`` or
``asyncio.to_thread`` each inherit the span that was open when they started and
never see each other's spans. Plain threads do not inherit context: pass
``parent_id`` explicitly there. Recording and checks are serialised by a lock.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import threading
import time
from collections.abc import Callable, Iterable
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from agent_eval.invariants import default_registry
from agent_eval.invariants.base import BaseInvariant, InvariantRegistry, InvariantViolation
from agent_eval.schemas import (
    AssertionResult,
    AttributeValue,
    EvalReport,
    Severity,
    Span,
    SpanKind,
    SpanStatus,
    Stage,
    Trajectory,
)

if TYPE_CHECKING:
    from agent_eval.telemetry.otel_exporter import TelemetryEmitter

logger = logging.getLogger("agent_eval")

Clock = Callable[[], int]
Invariants = InvariantRegistry | Iterable[BaseInvariant] | None

_current_session: contextvars.ContextVar[EvaluationSession | None] = contextvars.ContextVar(
    "agent_eval_session", default=None
)

# Open spans of the current execution context, as (session, span) pairs. A
# tuple is immutable, so a task that pushes a span never changes what its
# siblings see.
_open_spans: contextvars.ContextVar[tuple[tuple[EvaluationSession, Span], ...]] = (
    contextvars.ContextVar("agent_eval_open_spans", default=())
)

AGENT_ERROR_INVARIANT = "agent_completed_without_error"
StreamMode = Literal["buffer", "passthrough"]


def current_session() -> EvaluationSession | None:
    """The session active in this context, if any."""
    return _current_session.get()


class SimClock:
    """Deterministic clock for replays, benchmarks and tests.

    Starts at wall clock time (so exported spans have sane timestamps) and only
    moves when :meth:`advance` is called.
    """

    def __init__(self, start_ns: int | None = None) -> None:
        self._now = time.time_ns() if start_ns is None else start_ns

    def __call__(self) -> int:
        return self._now

    def advance(self, ms: float) -> None:
        if ms < 0:
            raise ValueError("cannot move the clock backwards")
        self._now += int(ms * 1_000_000)


def _as_registry(invariants: Invariants) -> InvariantRegistry:
    if invariants is None:
        return default_registry()
    if isinstance(invariants, InvariantRegistry):
        return invariants
    return InvariantRegistry(invariants)


class SpanScope:
    """Context manager for one live span. Returned by ``session.agent()`` etc."""

    def __init__(self, session: EvaluationSession, span: Span, explicit_parent: bool) -> None:
        self._session = session
        self._explicit_parent = explicit_parent
        self._token: contextvars.Token[Any] | None = None
        self.span = span

    def set_output(self, output: Any) -> SpanScope:
        self.span.output = output
        return self

    def set_tokens(self, input_tokens: int = 0, output_tokens: int = 0) -> SpanScope:
        self.span.input_tokens = input_tokens
        self.span.output_tokens = output_tokens
        return self

    def set_attribute(self, key: str, value: AttributeValue) -> SpanScope:
        self.span.attributes[key] = value
        return self

    def mark_handled(self) -> SpanScope:
        """Declare that a failure on this span was dealt with by the caller."""
        self.span.attributes["error.handled"] = True
        return self

    def __enter__(self) -> SpanScope:
        self._session._open(self.span, self._explicit_parent)
        self._token = _open_spans.set((*_open_spans.get(), (self._session, self.span)))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        blocked = isinstance(exc, InvariantViolation)
        if exc is not None and not blocked:
            self.span.status = SpanStatus.ERROR
            self.span.error = f"{type(exc).__name__}: {exc}"
        try:
            if self._token is not None:
                _open_spans.reset(self._token)
        except ValueError:
            # Closed from another context (for example a generator resumed
            # elsewhere): drop just this span.
            _open_spans.set(tuple(e for e in _open_spans.get() if e[1] is not self.span))
        self._token = None
        self._session._close(self.span, check=not blocked)
        return False


class EvaluationSession:
    """Collects a trajectory and enforces invariants on it.

    Example::

        with EvaluationSession("dispute", invariants=registry) as session:
            session.set_input(user_message)
            with session.agent("triage") as step:
                step.set_tokens(350, 60)
            with session.tool("ledger.get_transaction", input={"id": "TXN-1"}) as call:
                call.set_output(ledger.get("TXN-1"))
            session.set_output(answer)
        report = session.report
    """

    def __init__(
        self,
        name: str = "agent",
        *,
        invariants: Invariants = None,
        enforce: bool = True,
        emitter: TelemetryEmitter | None = None,
        clock: Clock | None = None,
        model: str | None = None,
        on_report: Callable[[EvalReport], None] | None = None,
    ) -> None:
        self.registry = _as_registry(invariants)
        self.enforce = enforce
        self.emitter = emitter
        self.clock: Clock = clock or time.time_ns
        self.model = model
        self.on_report = on_report
        self.trajectory = Trajectory(name=name, model=model)
        self.report: EvalReport | None = None
        self._lock = threading.RLock()
        self._token: contextvars.Token[EvaluationSession | None] | None = None
        self._prechecked_inputs: dict[str, Any] = {}
        self._emitted = False

    # ---- lifecycle --------------------------------------------------------------

    def __enter__(self) -> EvaluationSession:
        self.trajectory.start_ns = self.clock()
        self._token = _current_session.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        try:
            if exc is None:
                self.finalize()
            elif isinstance(exc, GeneratorExit):
                self._build_final_report()
            elif not isinstance(exc, InvariantViolation):
                # The agent crashed. Fail closed: the run cannot pass.
                self.trajectory.error = f"{type(exc).__name__}: {exc}"
                self._build_final_report()
        finally:
            if self._token is not None:
                try:
                    _current_session.reset(self._token)
                except ValueError:  # exited from another context
                    if _current_session.get() is self:
                        _current_session.set(None)
                self._token = None
            self._publish()
        return False

    @property
    def blocked(self) -> bool:
        return self.report is not None and not self.report.passed

    # ---- recording --------------------------------------------------------------

    def set_input(self, payload: Any) -> None:
        """Record the agent input and run input stage invariants."""
        with self._lock:
            self.trajectory.input = payload
            self._check(Stage.INPUT)

    def set_output(self, payload: Any) -> None:
        """Record the final output. It is evaluated by :meth:`finalize`."""
        self.trajectory.final_output = payload

    def span(
        self,
        kind: SpanKind,
        name: str,
        *,
        agent_name: str | None = None,
        tool_name: str | None = None,
        model: str | None = None,
        input: Any = None,  # noqa: A002 - mirrors the span field name
        parent_id: str | None = None,
        attributes: dict[str, AttributeValue] | None = None,
    ) -> SpanScope:
        span = Span(
            kind=kind,
            name=name,
            agent_name=agent_name,
            tool_name=tool_name,
            model=model,
            input=input,
            parent_id=parent_id,
            attributes=dict(attributes or {}),
        )
        return SpanScope(self, span, explicit_parent=parent_id is not None)

    def agent(
        self,
        name: str,
        *,
        model: str | None = None,
        input: Any = None,  # noqa: A002
        **kwargs: Any,
    ) -> SpanScope:
        """Open an ``invoke_agent`` span."""
        return self.span(
            SpanKind.AGENT,
            f"invoke_agent {name}",
            agent_name=name,
            model=model or self.model,
            input=input,
            **kwargs,
        )

    def tool(self, name: str, *, input: Any = None, **kwargs: Any) -> SpanScope:  # noqa: A002
        """Open an ``execute_tool`` span. Exceptions mark it failed and re-raise."""
        return self.span(
            SpanKind.TOOL,
            f"execute_tool {name}",
            tool_name=name,
            agent_name=self._enclosing_agent(),
            input=input,
            **kwargs,
        )

    def llm(self, model: str | None = None, *, input: Any = None, **kwargs: Any) -> SpanScope:  # noqa: A002
        """Open a ``chat`` span for a single model call."""
        model = model or self.model
        return self.span(
            SpanKind.LLM,
            f"chat {model}" if model else "chat",
            model=model,
            agent_name=self._enclosing_agent(),
            input=input,
            **kwargs,
        )

    def record_span(self, span: Span) -> Span:
        """Record a finished span, for example from a framework callback.

        A span without a parent is attached to the currently open span, if any.
        Span stage invariants run immediately.
        """
        stack = self._stack()
        if span.parent_id is None and stack:
            span.parent_id = stack[-1].span_id
        with self._lock:
            if span.start_ns == 0:
                span.start_ns = self.clock()
            if span.end_ns is None:
                span.end_ns = max(span.start_ns, self.clock())
            self.trajectory.spans.append(span)
            if not self.blocked:
                self._check(Stage.SPAN, span)
        return span

    # ---- evaluation -------------------------------------------------------------

    def finalize(self) -> EvalReport:
        """Evaluate every invariant on the complete trajectory.

        Raises ``InvariantViolation`` when enforcing and a blocking invariant
        failed. Safe to call more than once.
        """
        with self._lock:
            if self.report is None:
                self._build_final_report()
        assert self.report is not None
        if self.enforce and not self.report.passed and not self._emitted:
            raise InvariantViolation(self.report)
        return self.report

    def _build_final_report(self) -> None:
        if self.report is not None:
            return
        self.trajectory.end_ns = self.clock()
        report = self.registry.evaluate(self.trajectory, Stage.OUTPUT)
        if self.trajectory.error is not None:
            error_type = self.trajectory.error.split(":", 1)[0]
            report.results.append(
                AssertionResult(
                    invariant=AGENT_ERROR_INVARIANT,
                    passed=False,
                    severity=Severity.BLOCK,
                    stage=Stage.OUTPUT,
                    message=f"agent raised {error_type}; failing closed",
                )
            )
            report.passed = False
            report.blocked_at = Stage.OUTPUT
        self.report = report

    def _check(self, stage: Stage, span: Span | None = None) -> None:
        """Run one gate. ``span`` is the span that just closed, so content
        invariants scan only its payloads instead of the whole trajectory."""
        self.trajectory.end_ns = self.clock()
        report = self.registry.evaluate(self.trajectory, stage, span=span)
        if not report.passed and self.enforce:
            self.report = report
            raise InvariantViolation(report)
        # In monitor mode, or on a pass, the run continues and the final
        # report is built on the complete trajectory.
        self.trajectory.end_ns = None

    def _publish(self) -> None:
        if self._emitted or self.report is None:
            return
        self._emitted = True
        if self.trajectory.end_ns is None:
            self.trajectory.end_ns = self.clock()
        if self.on_report is not None:
            try:
                self.on_report(self.report)
            except Exception:  # noqa: BLE001 - callback errors must not break the verdict
                logger.exception("on_report callback failed for trace %s", self.trajectory.trace_id)
        if self.emitter is not None:
            try:
                self.emitter.emit(self.trajectory, self.report)
            except Exception:  # noqa: BLE001 - telemetry must not change the verdict
                logger.exception("telemetry export failed for trace %s", self.trajectory.trace_id)

    # ---- span stack -------------------------------------------------------------

    def _stack(self) -> list[Span]:
        """Spans of this session that are open in the current context."""
        return [span for session, span in _open_spans.get() if session is self]

    def _enclosing_agent(self) -> str | None:
        for span in reversed(self._stack()):
            if span.agent_name:
                return span.agent_name
        return None

    def _open(self, span: Span, explicit_parent: bool) -> None:
        stack = self._stack()
        if not explicit_parent and stack:
            span.parent_id = stack[-1].span_id
        with self._lock:
            span.start_ns = self.clock()
            self.trajectory.spans.append(span)
            if span.input is not None and not self.blocked:
                self._prechecked_inputs[span.span_id] = span.input
                try:
                    self._check(Stage.SPAN, span)
                except InvariantViolation:
                    span.end_ns = span.start_ns
                    raise

    def _close(self, span: Span, *, check: bool) -> None:
        with self._lock:
            span.end_ns = max(span.start_ns, self.clock())
            prechecked = self._prechecked_inputs.pop(span.span_id, None)
            if check and not self.blocked:
                focus = (
                    span.model_copy(update={"input": None})
                    if prechecked is not None and span.input is prechecked
                    else span
                )
                self._check(Stage.SPAN, focus)


def _capture_input(signature: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any],
                   session_kwarg: str) -> Any:
    try:
        bound = signature.bind_partial(*args, **kwargs).arguments
    except TypeError:
        return {"args": list(args), "kwargs": kwargs}
    data = {k: v for k, v in bound.items() if k != session_kwarg}
    if len(data) == 1:
        return next(iter(data.values()))
    return data


def _join_chunks(chunks: list[Any]) -> Any:
    """Final output of a streamed run: joined text, or the list of chunks."""
    if all(isinstance(chunk, str) for chunk in chunks):
        return "".join(chunks)
    return chunks


def evaluate_agent(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    invariants: Invariants = None,
    enforce: bool = True,
    emitter: TelemetryEmitter | None = None,
    clock: Clock | None = None,
    model: str | None = None,
    on_report: Callable[[EvalReport], None] | None = None,
    session_kwarg: str = "session",
    stream: StreamMode = "buffer",
) -> Any:
    """Wrap any agent entry point in an ``EvaluationSession``.

    Works on plain functions, ``async def`` functions, generator functions and
    async generator functions. The call arguments become the trajectory input
    and the return value (or the joined stream) becomes the final output. If
    the function declares a parameter named ``session`` the live session is
    passed in; otherwise use ``current_session()``.

    On a blocking violation the wrapper raises ``InvariantViolation`` instead
    of returning, so a non compliant output cannot reach the caller.

    Streaming (generator functions):

    * ``stream="buffer"`` (default, fail closed): the stream is consumed inside
      the session, the output gate runs, and only then are the chunks yielded.
      Nothing reaches the caller from a blocked run.
    * ``stream="passthrough"``: chunks are yielded as they are produced and the
      output gate runs when the stream ends. A violation is raised after the
      last chunk, so the text has already been released. Use it only where low
      latency matters more than blocking, together with input and span gates.

    Reports, in order of preference under concurrency:

    * ``result, report = wrapped.with_report(...)`` (plain and async functions),
    * ``on_report`` callback, called once per run,
    * ``InvariantViolation.report`` on a blocked run,
    * ``wrapped.last_report``: a convenience for single threaded use. It is one
      shared slot, so concurrent callers overwrite each other.
    """
    if stream not in ("buffer", "passthrough"):
        raise ValueError("stream must be 'buffer' or 'passthrough'")

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(func)
        wants_session = session_kwarg in signature.parameters
        registry = _as_registry(invariants)
        label = name or func.__name__

        def new_session(sink: Callable[[EvalReport], None] | None = None) -> EvaluationSession:
            def publish(report: EvalReport) -> None:
                wrapper.last_report = report  # type: ignore[attr-defined]
                if sink is not None:
                    sink(report)
                if on_report is not None:
                    on_report(report)

            return EvaluationSession(
                label,
                invariants=registry,
                enforce=enforce,
                emitter=emitter,
                clock=clock,
                model=model,
                on_report=publish,
            )

        def begin(session: EvaluationSession, args: tuple[Any, ...],
                  kwargs: dict[str, Any]) -> dict[str, Any]:
            session.set_input(_capture_input(signature, args, kwargs, session_kwarg))
            if wants_session:
                kwargs = {**kwargs, session_kwarg: session}
            return kwargs

        def reject_lazy_result(result: Any) -> None:
            """A lazy result would leave the session before any work happened."""
            if inspect.isgenerator(result) or inspect.isasyncgen(result):
                if inspect.isgenerator(result):
                    result.close()
                raise TypeError(
                    f"{label} returned a generator. Decorate the generator function itself "
                    "so the stream is evaluated."
                )
            if inspect.iscoroutine(result):
                result.close()
                raise TypeError(
                    f"{label} returned a coroutine from a plain function. Decorate the "
                    "async function itself so it is awaited inside the session."
                )

        wrapper: Callable[..., Any]

        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                chunks: list[Any] = []
                with new_session() as session:
                    kwargs = begin(session, args, kwargs)
                    try:
                        async for chunk in func(*args, **kwargs):
                            chunks.append(chunk)
                            if stream == "passthrough":
                                yield chunk
                    finally:
                        session.set_output(_join_chunks(chunks))
                if stream == "buffer":
                    for chunk in chunks:
                        yield chunk

        elif inspect.isgeneratorfunction(func):

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                chunks: list[Any] = []
                with new_session() as session:
                    kwargs = begin(session, args, kwargs)
                    try:
                        for chunk in func(*args, **kwargs):
                            chunks.append(chunk)
                            if stream == "passthrough":
                                yield chunk
                    finally:
                        session.set_output(_join_chunks(chunks))
                if stream == "buffer":
                    yield from chunks

        elif inspect.iscoroutinefunction(func):

            async def run_async(args: tuple[Any, ...], kwargs: dict[str, Any],
                                sink: Callable[[EvalReport], None] | None) -> Any:
                with new_session(sink) as session:
                    kwargs = begin(session, args, kwargs)
                    result = await func(*args, **kwargs)
                    reject_lazy_result(result)
                    session.set_output(result)
                return result

            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                return await run_async(args, kwargs, None)

            async def with_report_async(*args: Any, **kwargs: Any) -> tuple[Any, EvalReport]:
                reports: list[EvalReport] = []
                result = await run_async(args, kwargs, reports.append)
                return result, reports[0]

            wrapper.with_report = with_report_async  # type: ignore[attr-defined]

        else:

            def run_sync(args: tuple[Any, ...], kwargs: dict[str, Any],
                         sink: Callable[[EvalReport], None] | None) -> Any:
                with new_session(sink) as session:
                    kwargs = begin(session, args, kwargs)
                    result = func(*args, **kwargs)
                    reject_lazy_result(result)
                    session.set_output(result)
                return result

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return run_sync(args, kwargs, None)

            def with_report(*args: Any, **kwargs: Any) -> tuple[Any, EvalReport]:
                reports: list[EvalReport] = []
                result = run_sync(args, kwargs, reports.append)
                return result, reports[0]

            wrapper.with_report = with_report  # type: ignore[attr-defined]

        wrapper.last_report = None  # type: ignore[attr-defined]
        return wrapper

    if fn is not None:
        return decorate(fn)
    return decorate
