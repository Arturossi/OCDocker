---
title: "refactor: Formatting-only cleanup for collapsed long lines"
status: completed
date: 2026-05-29
type: refactor
branch: new_protocol
---

# refactor: Formatting-only cleanup for collapsed long lines

## Summary

Expand collapsed physical lines in seven priority files so diffs are reviewable, without changing dependency semantics, config values, docking logic, or OCScore behavior. Most config/metadata files on `new_protocol` are already readable; the dominant risk is `OCDocker/Config.py` (a ~6,850-character default list literal).

## Problem Frame

Several important files are hard to review because content is compressed into very long lines. This blocks dependency auditing, config review, and Python maintenance. The user explicitly forbids dependency reorganization, version changes, API changes, and scientific/protocol behavior changes in this pass.

**Current state (branch audit):**

| File | Max line length | Notes |
|------|---------------:|-------|
| `pyproject.toml` | ~122 | Already multiline arrays; verify-only |
| `requirements.txt` | ~25 | Already one dep per line; verify-only |
| `pytest.ini` | ~101 | Readable; verify-only |
| `.pre-commit-config.yaml` | ~80 | Valid block YAML; verify-only |
| `OCDocker/Config.py` | **6,850** | `reference_column_order` default inline list |
| `OCDocker/Docking/BaseVinaLike.py` | ~147 | Long function signatures |
| `OCDockerConsole.py` | ~124 | License prose line |

**Out of scope (explicit):** dependency moves between core/extras, version bumps, `Initialise.py`, `Ligand.py`, `OCDocker.cfg.example`, and any file not listed above.

---

## Requirements

- R1. All seven listed files use readable formatting (multiline TOML/YAML/INI; Black-compatible Python at line-length 120).
- R2. Parsed metadata/config values are byte-identical to pre-format state (dependencies, versions, pytest options, pre-commit hooks, default config lists).
- R3. No public API, CLI, docking, or OCScore behavior changes.
- R4. Verification runs after formatting: TOML/YAML/INI parse checks, Python compile/import checks, targeted pytest subset.
- R5. Deliverable includes an exact list of formatting-only changed files.

---

## Key Technical Decisions

- KTD1: **Verify-first on metadata files** — `pyproject.toml`, `requirements.txt`, `pytest.ini`, and `.pre-commit-config.yaml` are already expanded on `new_protocol`; only touch them if audit finds compressed lines. Avoid churn-only reformats.
- KTD2: **`Config.py` default list extraction** — Move the inline `reference_column_order` default to a module-level constant (same pattern as `GNINA_DEFAULT_SCORING_FUNCTIONS`) with one string per line; reference it from `cfg.get(...)`. Preserves exact element order and spelling; improves reviewability without semantic change.
- KTD3: **Python formatter** — Run `black --line-length 120` on the three Python targets only; do not run repo-wide Black. Ruff docstring rules remain unchanged.
- KTD4: **Equivalence proof for Config default** — Before/after compare of the default column list (length + ordered tuple) in verification; optional lightweight test if none exists.
- KTD5: **No mechanical dependency sorting** — Do not alphabetize extras or reorder TOML sections beyond expanding collapsed lines.

---

## Scope Boundaries

### In scope

- Formatting-only edits to the seven listed files.
- Module-level constant extraction in `Config.py` when it is the minimal way to break a 6k+ character line without changing values.

### Deferred to Follow-Up Work

- Formatting `OCDocker/Initialise.py`, `OCDocker/Ligand.py`, `OCDocker.cfg.example` (similar collapsed lines; separate PR).
- Dependency group reorganization (already addressed elsewhere; not part of this pass).
- Adding Black/ruff-format to pre-commit (tooling change, not formatting-only).

### Non-goals

- Refactoring config architecture, splitting modules, or renaming public symbols.
- Changing `reference_column_order` content or order.

---

## Implementation Units

### U1. Audit metadata and config files (verify-only baseline)

**Goal:** Confirm four non-Python targets already meet acceptance criteria; document baseline checksums.

**Requirements:** R1, R2, R4

**Files:** `pyproject.toml`, `requirements.txt`, `pytest.ini`, `.pre-commit-config.yaml`

**Approach:**

1. Record max line length per file.
2. Parse: `tomllib.load`, `yaml.safe_load`, `configparser` for pytest.ini.
3. If any line exceeds ~200 chars or arrays are inline-compressed, expand per user rules without changing values.
4. If already compliant, leave unchanged and note in commit message.

**Patterns to follow:** Current `pyproject.toml` multiline `[project.optional-dependencies]` blocks.

**Test scenarios:**

- Happy path: TOML loads; core dep list matches `requirements.txt` (existing `tests/cli/test_packaging_metadata.py`).
- Edge case: `gpu` extra retains `sys_platform != "darwin"` marker verbatim after any TOML touch.

**Verification:** Parse succeeds; `pytest tests/cli/test_packaging_metadata.py -q` passes; no diff in dependency names/versions if file untouched.

---

### U2. Expand `Config.py` collapsed default list

**Goal:** Eliminate the ~6,850-character line while preserving identical default `reference_column_order`.

**Requirements:** R1, R2, R3

**Dependencies:** U1 (no pyproject coupling; sequencing for commit hygiene only)

**Files:** `OCDocker/Config.py`, `tests/core/test_config.py` (optional assertion add)

**Approach:**

1. Capture pre-change default list via one-liner import or AST parse of the inline literal.
2. Add `DEFAULT_REFERENCE_COLUMN_ORDER: List[str] = [...]` near existing `GNINA_DEFAULT_*` constants; one column name per line, trailing commas.
3. Replace `cfg.get('reference_column_order', [...])` with `cfg.get('reference_column_order', DEFAULT_REFERENCE_COLUMN_ORDER.copy())` or equivalent that preserves mutability semantics of the prior inline list default.
4. Run `black --line-length 120 OCDocker/Config.py`.
5. Re-compare ordered list equality.

**Execution note:** Characterization-first — capture the ordered default list before editing; fail verification if order or length changes.

**Patterns to follow:** `GNINA_DEFAULT_SCORING_FUNCTIONS` / `GNINA_DEFAULT_CNN_MODELS` list constants in the same file.

**Test scenarios:**

- Happy path: `OCDockerConfig.from_dict({})` or minimal config load yields same `paths.reference_column_order` length and first/last elements as pre-change snapshot.
- Edge path: `cfg.get` override still wins when `reference_column_order` present in parsed config (`tests/core/test_initialise_config_parser.py` pattern).
- Error path: none (formatting-only).

**Verification:** Ordered list byte-for-byte equal; `pytest tests/core/test_config.py -q` passes.

---

### U3. Format `BaseVinaLike.py` and `OCDockerConsole.py`

**Goal:** Black-compatible readability for remaining long Python lines.

**Requirements:** R1, R3

**Dependencies:** U2 (independent; may parallelize)

**Files:** `OCDocker/Docking/BaseVinaLike.py`, `OCDockerConsole.py`, `tests/docking/test_smina_utilities.py` (if signatures affect import paths — expect no test changes)

**Approach:**

1. Run `black --line-length 120` on both files only.
2. Manually inspect diff: signature wraps, license block wraps, no logic edits.
3. `python -m compileall` on both modules.

**Patterns to follow:** Project `pyproject.toml` `[tool.ruff] line-length = 120`.

**Test scenarios:**

- Happy path: modules import without error.
- Integration: `pytest tests/docking/test_smina_utilities.py -q` (uses `BaseVinaLike` via Smina paths).

**Verification:** `compileall` clean; targeted docking tests pass; no diff outside whitespace/wrapping.

---

### U4. Verification gate and change summary

**Goal:** Prove formatting-only outcome and produce file list for PR description.

**Requirements:** R4, R5

**Dependencies:** U1, U2, U3

**Files:** (verification only)

**Approach:**

Run verification bundle:

```text
python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"
python -c "import yaml; yaml.safe_load(open('.pre-commit-config.yaml'))"
python -m compileall OCDocker/Config.py OCDocker/Docking/BaseVinaLike.py OCDockerConsole.py
pytest tests/cli/test_packaging_metadata.py tests/core/test_config.py tests/docking/test_smina_utilities.py -q
```

Optional: `git diff --stat` limited to the seven files; assert no other paths modified.

**Test scenarios:**

- Happy path: all commands exit 0.
- Regression: packaging metadata tests still align `requirements.txt` with core deps.

**Verification:** Checklist complete; PR summary lists only formatting-touched files with note for verify-only unchanged files.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Accidental reorder of `reference_column_order` breaks OCScore masking | Pre/post ordered tuple compare; run `tests/core/test_config.py` |
| `black` reformats beyond wrapping | Scope `black` to three files; review diff |
| Touching already-good `pyproject.toml` creates noise | U1 verify-first; skip if compliant |
| Default list mutability (`list` shared reference) | Use `.copy()` on constant if prior code relied on fresh list per call |

---

## Acceptance Checklist

- [ ] `pyproject.toml` has multiline arrays (no compressed mega-lines)
- [ ] `requirements.txt` one requirement per line
- [ ] `pytest.ini` readable standard INI
- [ ] `.pre-commit-config.yaml` valid, readable YAML
- [ ] Python files Black-compatible at line-length 120
- [ ] No behavior, API, dependency version, or protocol changes
- [ ] Verification bundle green
- [ ] Exact formatting-only file list documented

---

## Sources and Research

- Branch file audit (2026-05-29): line-length survey on target paths
- Existing tests: `tests/cli/test_packaging_metadata.py`, `tests/core/test_config.py`, `tests/docking/test_smina_utilities.py`
- Prior packaging work on `new_protocol` already expanded `pyproject.toml`; this plan treats metadata files as verify-first
