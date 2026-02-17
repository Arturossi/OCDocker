#!/usr/bin/env python3

# Description
###############################################################################
'''
Packaging metadata consistency tests.

These tests keep pyproject optional extras and requirements.txt aligned.
'''

# Imports
###############################################################################
import tomllib

from pathlib import Path

import pytest

# Functions
###############################################################################
## Private ##


def _load_pyproject() -> dict:
    with Path("pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def _load_requirements_lines() -> list[str]:
    lines: list[str] = []
    for raw in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


## Public ##


@pytest.mark.order(468)
def test_pyproject_uses_ml_extra_name():
    data = _load_pyproject()
    extras = data["project"]["optional-dependencies"]

    assert "ml" in extras
    assert "optuna" not in extras


@pytest.mark.order(469)
def test_requirements_match_core_pyproject_dependencies():
    data = _load_pyproject()
    pyproject_core = data["project"]["dependencies"]
    requirements_core = _load_requirements_lines()

    assert requirements_core == pyproject_core


@pytest.mark.order(470)
def test_ml_extra_contains_ml_runtime_stack():
    data = _load_pyproject()
    ml_extra = data["project"]["optional-dependencies"]["ml"]

    expected_prefixes = (
        "torch",
        "torchaudio",
        "torchvision",
        "xgboost",
        "visualtorch",
        "optuna",
        "optuna-dashboard",
        "optuna-integration",
    )
    for prefix in expected_prefixes:
        assert any(dep.startswith(prefix) for dep in ml_extra)

    assert "torchsummary" in ml_extra
    assert "torchviz" in ml_extra


@pytest.mark.order(471)
def test_core_dependencies_exclude_ml_runtime_stack():
    data = _load_pyproject()
    pyproject_core = data["project"]["dependencies"]
    excluded_prefixes = (
        "torch",
        "torchaudio",
        "torchvision",
        "xgboost",
        "visualtorch",
        "optuna",
        "optuna-dashboard",
        "optuna-integration",
    )

    for prefix in excluded_prefixes:
        assert not any(dep.startswith(prefix) for dep in pyproject_core)

    assert "torchsummary" not in pyproject_core
    assert "torchviz" not in pyproject_core
