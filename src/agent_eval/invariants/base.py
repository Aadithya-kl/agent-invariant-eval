"""Invariant base class and the fail-closed registry.

Fail closed means that anything other than a clean, explicit pass is a failure:

* an invariant that raises is recorded as a failed assertion,
* an invariant that returns something other than an ``AssertionResult`` fails,
* an empty registry fails, because "nothing was checked" must never read as
  "everything passed".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import ClassVar

from agent_eval.schemas import AssertionResult, EvalReport, Severity, Span, Stage, Trajectory

ALL_STAGES: frozenset[Stage] = frozenset(Stage)


class InvariantViolation(Exception):
    """Raised when a blocking invariant fails. Carries the full report."""

    def __init__(self, report: EvalReport) -> None:
        self.report = report
        details = "; ".join(f"{r.invariant}: {r.message}" for r in report.violations)
        stage = report.blocked_at.value if report.blocked_at else report.stage.value
        super().__init__(f"[{report.name}] blocked at {stage} stage: {details}")


class BaseInvariant(ABC):
    """A deterministic, synchronous assertion over a trajectory.

    Subclasses implement :meth:`check` as a pure function of the trajectory.
    The same invariant is evaluated on partial trajectories during execution
    (``Stage.INPUT`` and ``Stage.SPAN``) and on the complete one at the end
    (``Stage.OUTPUT``), so ``check`` must tolerate missing spans and outputs.
    """

    name: ClassVar[str]
    stages: ClassVar[frozenset[Stage]] = ALL_STAGES

    def __init__(self, *, severity: Severity = Severity.BLOCK) -> None:
        self.severity = severity

    @abstractmethod
    def check(self, trajectory: Trajectory, stage: Stage) -> AssertionResult:
        """Return a pass or fail result. May raise; the caller fails closed."""

    def check_span(self, trajectory: Trajectory, span: Span, stage: Stage) -> AssertionResult:
        """Span stage check when the span that just closed is known.

        The default re-checks the whole trajectory. Content invariants override
        this to look only at ``span``, which keeps the cost of a run linear in
        the number of spans. The output stage always runs the full
        :meth:`check`, so nothing is skipped in the final verdict.
        """
        return self.check(trajectory, stage)

    # ---- helpers for subclasses -------------------------------------------------

    def ok(self, stage: Stage, message: str = "", **metrics: float) -> AssertionResult:
        return AssertionResult(
            invariant=self.name,
            passed=True,
            severity=self.severity,
            stage=stage,
            message=message,
            metrics=metrics,
        )

    def fail(
        self,
        stage: Stage,
        message: str,
        *,
        evidence: Iterable[str] = (),
        span_ids: Iterable[str] = (),
        **metrics: float,
    ) -> AssertionResult:
        return AssertionResult(
            invariant=self.name,
            passed=False,
            severity=self.severity,
            stage=stage,
            message=message,
            evidence=list(evidence),
            span_ids=list(dict.fromkeys(span_ids)),
            metrics=metrics,
        )

    # ---- execution --------------------------------------------------------------

    def evaluate(
        self,
        trajectory: Trajectory,
        stage: Stage = Stage.OUTPUT,
        *,
        span: Span | None = None,
    ) -> AssertionResult:
        """Run the check and convert any error into a failed assertion.

        ``span`` is the span that just closed. It is only used at the span stage.
        """
        try:
            if span is not None and stage is Stage.SPAN:
                result = self.check_span(trajectory, span, stage)
            else:
                result = self.check(trajectory, stage)
        except Exception as exc:  # noqa: BLE001 - fail closed on any error
            return AssertionResult(
                invariant=self.name,
                passed=False,
                severity=Severity.BLOCK,
                stage=stage,
                message=f"invariant raised {type(exc).__name__}; failing closed",
            )
        if not isinstance(result, AssertionResult):
            return AssertionResult(
                invariant=self.name,
                passed=False,
                severity=Severity.BLOCK,
                stage=stage,
                message="invariant returned a non AssertionResult value; failing closed",
            )
        return result

    def enforce(self, target: Trajectory | str, stage: Stage = Stage.OUTPUT) -> AssertionResult:
        """Assert style entry point. Raises :class:`InvariantViolation` on failure.

        ``target`` may be a trajectory or a plain string, which is treated as a
        final output with no tool calls behind it.
        """
        trajectory = target if isinstance(target, Trajectory) else Trajectory(final_output=target)
        result = self.evaluate(trajectory, stage)
        if not result.passed:
            raise InvariantViolation(
                EvalReport(
                    trace_id=trajectory.trace_id,
                    name=trajectory.name,
                    model=trajectory.model,
                    stage=stage,
                    passed=False,
                    blocked_at=stage,
                    results=[result],
                    metrics=trajectory.metrics(),
                )
            )
        return result

    __call__ = enforce


class InvariantRegistry:
    """Ordered, fail-closed collection of invariants."""

    EMPTY_SENTINEL = "registry_not_empty"

    def __init__(self, invariants: Iterable[BaseInvariant] = ()) -> None:
        self._invariants: dict[str, BaseInvariant] = {}
        for invariant in invariants:
            self.register(invariant)

    def register(self, invariant: BaseInvariant) -> InvariantRegistry:
        if not isinstance(invariant, BaseInvariant):
            raise TypeError(f"expected a BaseInvariant, got {type(invariant).__name__}")
        name = getattr(invariant, "name", None)
        if not name:
            raise ValueError(f"{type(invariant).__name__} does not define a name")
        if name in self._invariants:
            raise ValueError(f"invariant {name!r} is already registered")
        if Stage.OUTPUT not in invariant.stages:
            raise ValueError(f"invariant {name!r} must run at the output stage")
        self._invariants[name] = invariant
        return self

    def get(self, name: str) -> BaseInvariant:
        try:
            return self._invariants[name]
        except KeyError:
            known = ", ".join(sorted(self._invariants)) or "none"
            raise KeyError(f"unknown invariant {name!r}; registered: {known}") from None

    def names(self) -> list[str]:
        return list(self._invariants)

    def __iter__(self) -> Iterator[BaseInvariant]:
        return iter(self._invariants.values())

    def __len__(self) -> int:
        return len(self._invariants)

    def __contains__(self, name: object) -> bool:
        return name in self._invariants

    def run(
        self,
        trajectory: Trajectory,
        stage: Stage = Stage.OUTPUT,
        *,
        span: Span | None = None,
    ) -> list[AssertionResult]:
        if not self._invariants:
            return [
                AssertionResult(
                    invariant=self.EMPTY_SENTINEL,
                    passed=False,
                    severity=Severity.BLOCK,
                    stage=stage,
                    message="no invariants registered; failing closed",
                )
            ]
        return [
            invariant.evaluate(trajectory, stage, span=span)
            for invariant in self._invariants.values()
            if stage in invariant.stages
        ]

    def evaluate(
        self,
        trajectory: Trajectory,
        stage: Stage = Stage.OUTPUT,
        *,
        span: Span | None = None,
    ) -> EvalReport:
        results = self.run(trajectory, stage, span=span)
        passed = not any(not r.passed and r.severity is Severity.BLOCK for r in results)
        return EvalReport(
            trace_id=trajectory.trace_id,
            name=trajectory.name,
            model=trajectory.model,
            stage=stage,
            passed=passed,
            blocked_at=None if passed else stage,
            results=results,
            metrics=trajectory.metrics(),
        )
