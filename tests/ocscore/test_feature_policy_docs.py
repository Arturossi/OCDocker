#!/usr/bin/env python3

"""Documentation consistency checks for OCScore feature policies."""

from __future__ import annotations

import re
from pathlib import Path


DOC_PATHS = [
    Path("README.md"),
    Path("OCSCORE_REPLICATION.md"),
    Path("docs/ocscore-production-protocol.md"),
]


def _docs_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in DOC_PATHS)


def _shell_blocks(text: str) -> list[str]:
    return re.findall(r"```(?:bash|shell|sh)\n(.*?)```", text, flags=re.DOTALL)


def test_feature_policy_docs_include_required_examples():
    text = _docs_text()

    assert "OCDocker/OCScore/Protocols/Ablations/" in text
    assert ".yml" in text
    assert "--feature-policy no_pmi" in text
    assert "--run-all-feature-policies" in text
    assert "--feature-policy-dir" in text
    assert "--feature-policy-yml" in text


def test_feature_policy_docs_do_not_require_yaml_extension():
    text = _docs_text()
    assert ".yml files, not .yaml" in text or "use `.yml` files, not `.yaml`" in text
    assert "required extension is .yaml" not in text


def test_docs_do_not_show_deprecated_or_reduced_training_commands():
    text = _docs_text()
    command_blocks = "\n".join(_shell_blocks(text))

    assert "--reduction-archive" not in command_blocks
    assert "--raw-input-dir" in command_blocks
    assert "selected_features.json" not in command_blocks
    assert "reduced_pdbbind.csv" not in command_blocks
    assert "reduced_dudez.csv" not in command_blocks
