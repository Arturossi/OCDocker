# Contributing to OCDocker

Thanks for your interest in contributing to OCDocker. This guide covers the
basics for proposing changes, setting up a dev environment, and running tests.

## Code of Conduct
Participation in this project is governed by `CODE_OF_CONDUCT.md`.

## Ways to Contribute
- Report bugs with clear, reproducible steps.
- Improve documentation and examples.
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
# Pip from source
pip install -r requirements.txt
pip install -e .
```

Helpful environment settings:
- `OCDOCKER_USE_SQLITE=1` to bypass MySQL in local dev/tests.
- `OCDOCKER_CONFIG=./OCDocker.cfg` if you want a custom config. Start from
  `OCDocker.cfg.example`.

## Tests
```bash
pytest -q
```

Notes:
- Tests use fixtures in `test_files/` and do not require external docking
  binaries to run.
- End-to-end docking requires tools such as MGLTools and engines like Vina,
  Smina, or PLANTS, with paths configured in `OCDocker.cfg`.

## Documentation
Documentation sources live in `docs/`. Update docs when you change user-facing
behavior or the CLI.

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
