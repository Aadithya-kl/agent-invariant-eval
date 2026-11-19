# Changelog

## 0.2.0

Addresses an external review of 0.1.0. Behaviour changes are marked **changed**.

### Runtime and concurrency

* Open span stack is now context local (`contextvars`). Parallel branches under
  `asyncio.gather`, `TaskGroup` and `to_thread` attach to the correct parent. Recording and
  checks are serialised by a lock, so threads can record safely with an explicit `parent_id`.
* PII and PCI scanning is incremental. A span boundary scans only the span that closed
  (`BaseInvariant.check_span`). The output stage still scans everything once.
* `@evaluate_agent` supports generator and async generator agents with `stream="buffer"`
  (default, fail closed) and `stream="passthrough"`. A plain function that returns a generator
  or coroutine object now fails closed with a `TypeError`.
* `wrapped.with_report(...)` returns `(result, report)` and is race free. `last_report` stays
  as a single threaded convenience.

### Invariants

* **changed** `numeric_grounding`: numbers inside ids, dates and times in tool strings no
  longer ground a figure. New `ground_from_strings` option. More currencies (symbols, ISO
  codes in upper case, Indian digit grouping).
* **changed** `no_pii`: a bare nine digit number is a routing number only with a routing
  keyword and a valid ABA checksum. `require_routing_context=False` restores checksum only.
  Context keywords now match keys such as `routing_number` and `ssn_number`.
* **changed** `no_context_bloat`: baseline floored at `min_baseline_tokens` (500).
* **changed** `no_swallowed_tool_failures`: the default acknowledgement pattern accepts
  inability phrases only. Apologies and bare words such as "error" no longer count. New
  `allow_acknowledgement` option.
* **changed** `no_routing_loops`: counts repeated directed handoffs (`max_edge_repeats`, default
  1). `max_visits_per_agent` is now optional and off by default. Hub and spoke supervisors pass.
* **changed** Token totals deduplicate rolled up parent usage
  (`Trajectory.token_attribution()`). SLA budgets, context bloat and costs use it.

### Telemetry, benchmark, CLI

* OTel trace and span ids now equal the trajectory ids. Every child span carries
  `agent_eval.span_id`, and `agent_eval.violations` when an invariant failed on it.
  **changed** root attribute `gen_ai.conversation.id` replaced by `agent_eval.trace_id`
  (a trace id is not a conversation id).
* `FrontierRunner` accepts async agents, `max_concurrency`, and `await runner.arun(...)`.
* `ModelConfig.pricing` prices each span by its model for mixed model runs.
* `agent-eval benchmark --models` accepts a JSON file or `module:attribute`. Added
  `--max-concurrency`. `examples/payment_dispute_agent/models.json` is now shipped.
* CLI tests added.

### Follow up fixes

* Abandoning a `passthrough` stream early evaluates the partial output instead of recording an
  agent crash.
* Token roll up is detected per side (input and output), which can only over count.
* Stricter digit grouping. Money in an unsupported number format (`€12,50`, `1.234,56 EUR`)
  now fails `numeric_grounding` as an unparseable figure instead of being skipped.

## 0.1.0

Initial release.
