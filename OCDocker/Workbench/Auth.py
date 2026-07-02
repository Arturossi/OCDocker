#!/usr/bin/env python3

# Description
###############################################################################
"""
Bearer-token authentication for execute-capable Workbench API endpoints.
"""

# Imports
###############################################################################
from __future__ import annotations

import os
import secrets

from pathlib import Path

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Constants
###############################################################################

WORKBENCH_JOB_TOKEN_ENV_VAR = "OCDOCKER_WORKBENCH_TOKEN"
WORKBENCH_JOB_TOKEN_FILENAME = "workbench_token"

# Functions
###############################################################################
## Private ##


def _config_dir() -> Path:
    '''Return the OCDocker user config directory, honoring XDG_CONFIG_HOME.

    Returns
    -------
    pathlib.Path
        OCDocker user config directory.
    '''

    xdg_home = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg_home) if xdg_home else Path.home() / ".config"
    return base / "ocdocker"


## Public ##


def workbench_job_token_path() -> Path:
    '''Return the on-disk path used to persist the Workbench job token.

    Returns
    -------
    pathlib.Path
        Workbench job token file path.
    '''

    return _config_dir() / WORKBENCH_JOB_TOKEN_FILENAME


def resolve_workbench_job_token(*, create: bool = True) -> str:
    '''Resolve the bearer token that gates Workbench job-execute endpoints.

    Resolution order: the ``OCDOCKER_WORKBENCH_TOKEN`` environment variable,
    then the token file, then (when ``create`` is True) a freshly generated
    token persisted to the token file.

    Parameters
    ----------
    create : bool
        Generate and persist a new token when none is configured yet.

    Returns
    -------
    str
        Resolved bearer token.

    Raises
    ------
    FileNotFoundError
        If no token is configured and ``create`` is False.
    '''

    env_value = os.environ.get(WORKBENCH_JOB_TOKEN_ENV_VAR, "").strip()
    if env_value:
        return env_value

    token_path = workbench_job_token_path()
    if token_path.is_file():
        stored = token_path.read_text(encoding="utf-8").strip()
        if stored:
            return stored

    if not create:
        raise FileNotFoundError(
            f"No Workbench job token configured. Set {WORKBENCH_JOB_TOKEN_ENV_VAR} "
            f"or create {token_path}."
        )

    token = secrets.token_urlsafe(32)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token + "\n", encoding="utf-8")
    token_path.chmod(0o600)
    return token


__all__ = [
    "WORKBENCH_JOB_TOKEN_ENV_VAR",
    "WORKBENCH_JOB_TOKEN_FILENAME",
    "resolve_workbench_job_token",
    "workbench_job_token_path",
]
