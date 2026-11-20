# Contributing

Contributions are welcome! Please keep changes focused, deterministic (no LLM calls in core
invariants), and covered by tests.

## Local setup

```bash
git clone https://github.com/sankaranainar-p/agent-invariant-eval.git
cd agent-invariant-eval
pip install -e ".[dev]"
```

## Checks before opening a PR

Run the same checks that CI runs:

```bash
ruff check src tests examples
mypy src
pytest --cov=agent_eval --cov-fail-under=90
python examples/payment_dispute_agent/run_eval.py
```

## Security issues

Do not open public issues for security vulnerabilities or detection bypasses. See
[SECURITY.md](SECURITY.md) for private reporting instructions.
