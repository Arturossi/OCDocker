# Contributing to OCDocker

Thanks for your interest in contributing to OCDocker. This guide covers the
basics for proposing changes, setting up a dev environment, and running tests.

## Code of Conduct
Participation in this project is governed by `CODE_OF_CONDUCT.md`.

## Ways to Contribute
- Report bugs with clear, reproducible steps.
- Improve documentation and examples (OCScore CLI changes: update [OCSCORE_REPLICATION.md](OCSCORE_REPLICATION.md)).
- Propose code changes with tests.

## Before You Start
- For large or user-facing changes, open an issue first so we can align on
  scope and approach.
- Keep pull requests focused on a single topic when possible.

## Development Setup
System dependencies are required for full functionality. See `README.md` for the
complete list and `install.sh` for an automated setup.

Choose one setup path:

```bash
# Conda
conda env create -f environment.yml
conda activate ocdocker
```

```bash
# Pip from source (minimal core)
pip install -r requirements.txt
pip install -e .

# Typical developer install with runtime stacks and test tooling
pip install -e ".[all,dev]"
```

Helpful environment settings:
- `OCDOCKER_DB_BACKEND=sqlite` to use local SQLite in dev/tests.
- `OCDOCKER_CONFIG=./OCDocker.cfg` if you want a custom config. Start from
  `OCDocker.cfg.example`.

## Tests
```bash
pytest -q
```

Good tests assert behavior users depend on (CLI output shape, metric values, security blocks, protocol invariants). Avoid tests that only lock package `__all__` or import/delegation wiring. See `obsidian/Architecture/Testing Map.md` for the fruitfulness rubric and protected suites.

Notes:
- Tests use fixtures in `test_files/` and do not require external docking
  binaries to run.
- End-to-end docking requires tools such as MGLTools and engines like Vina,
  Smina, or PLANTS, with paths configured in `OCDocker.cfg`.

## Documentation
Documentation sources live in `docs/`. Update docs when you change user-facing
behavior or the CLI.

Formatting and readability standards (line length 120, multiline TOML arrays,
module-level constants for long Python lists) are documented in
[`docs/source/development.rst`](docs/source/development.rst) (Sphinx: *Development conventions*).

## Output and Logging

OCDocker supports structured logging via `OCDocker.Toolbox.Logging`. Prefer this
for **library code** so callers (CLI, notebooks, tests) can control verbosity.

- **Library modules**: use a module-level logger and log at appropriate levels:

```python
import OCDocker.Toolbox.Logging as oclogging

LOGGER = oclogging.get_logger("ocscore.feature_reduction")
LOGGER.info("message")
LOGGER.warning("message")
LOGGER.error("message")
```

- **CLI / interactive UX**: it is acceptable to use direct `print()` for
  user-facing guidance (installation hints, prompts, final reports meant for
  piping). Prefer logging for diagnostics, retries, and debugging output.
  Keep `print()` out of library modules unless there is a strong UX reason.

- **Logging configuration**: configure logging once at the CLI boundary (or
  entrypoint) via `oclogging.configure(...)`. Avoid configuring logging deep
  inside library modules.

## Submitting Changes
- Use a dedicated branch.
- Include a clear summary, motivation, and test results in the PR.
- Add or update tests where feasible.
- Avoid committing large binaries or generated artifacts.

## Credit
If your contribution is accepted, add your name to `COLLABORATORS.md` in the
same pull request.

## License
By contributing, you agree that your contributions are licensed under the terms
in `LICENSE`.
