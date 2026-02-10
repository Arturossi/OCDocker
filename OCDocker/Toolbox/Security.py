#!/usr/bin/env python3

# Description
###############################################################################
'''
Security helpers for operations that cross trust boundaries.

Usage:

import OCDocker.Toolbox.Security as ocsec
'''

# Imports
###############################################################################
import os

from typing import Optional

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

## Public ##

ENV_ALLOW_UNSAFE_DESERIALIZATION = "OCDOCKER_ALLOW_UNSAFE_DESERIALIZATION"
ENV_ALLOW_SCRIPT_EXEC = "OCDOCKER_ALLOW_SCRIPT_EXEC"


def allow_unsafe_runtime(
    *,
    deserialization: bool = True,
    script_exec: bool = True,
) -> None:
    '''Allow unsafe runtime operations in the current process.

    This function sets opt-in environment flags used by security gates in
    OCDocker. Intended for trusted internal scripts/workflows.

    Parameters
    ----------
    deserialization : bool, optional
        If True, enables pickle/joblib/torch deserialization gates.
        Default is True.
    script_exec : bool, optional
        If True, enables dynamic script execution gate.
        Default is True.
    '''

    if deserialization:
        os.environ[ENV_ALLOW_UNSAFE_DESERIALIZATION] = "1"
    if script_exec:
        os.environ[ENV_ALLOW_SCRIPT_EXEC] = "1"

def env_flag_enabled(env_var: str) -> bool:
    '''Check whether an environment variable is set to a truthy value.

    Parameters
    ----------
    env_var : str
        Name of the environment variable to inspect.

    Returns
    -------
    bool
        True when environment variable is set to one of:
        1, true, yes, y, on (case-insensitive).
    '''

    value = os.getenv(env_var, "")
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def require_trusted_input(
    *,
    trusted: bool,
    operation: str,
    env_var: str,
    source: Optional[str] = None,
) -> None:
    '''Enforce explicit trust for high-risk operations.

    Parameters
    ----------
    trusted : bool
        Explicit opt-in from caller that input is trusted.
    operation : str
        Human-readable operation description (e.g., "pickle deserialization").
    env_var : str
        Environment variable that can globally opt in.
    source : str, optional
        Optional input source path shown in diagnostic message.

    Raises
    ------
    PermissionError
        If neither explicit trust nor environment opt-in is set.
    '''

    if trusted or env_flag_enabled(env_var):
        return

    source_str = f" for '{source}'" if source else ""
    raise PermissionError(
        f"Blocked {operation}{source_str}. This operation can execute arbitrary code with "
        f"untrusted inputs. Pass trusted=True, call allow_unsafe_runtime(), or set {env_var}=1."
    )
