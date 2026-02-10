#!/usr/bin/env python3

# Description
###############################################################################
'''
Unit tests for Toolbox security helpers.
'''

# Imports
###############################################################################
import pytest

import OCDocker.Toolbox.Security as ocsec

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

@pytest.mark.order(72)
def test_env_flag_enabled_truthy_and_falsey(monkeypatch):
    env_var = "OCDOCKER_TEST_SECURITY_FLAG"

    for value in ("1", "true", "yes", "y", "on", " TRUE "):
        monkeypatch.setenv(env_var, value)
        assert ocsec.env_flag_enabled(env_var) is True

    for value in ("0", "false", "no", "off", "", "random"):
        monkeypatch.setenv(env_var, value)
        assert ocsec.env_flag_enabled(env_var) is False

    monkeypatch.delenv(env_var, raising=False)
    assert ocsec.env_flag_enabled(env_var) is False


@pytest.mark.order(73)
def test_require_trusted_input_allows_explicit_trust(monkeypatch):
    monkeypatch.delenv(ocsec.ENV_ALLOW_UNSAFE_DESERIALIZATION, raising=False)

    # Should not raise when caller explicitly marks input as trusted
    ocsec.require_trusted_input(
        trusted=True,
        operation="pickle deserialization",
        env_var=ocsec.ENV_ALLOW_UNSAFE_DESERIALIZATION,
        source="trusted.pkl",
    )


@pytest.mark.order(74)
def test_require_trusted_input_allows_env_opt_in(monkeypatch):
    monkeypatch.setenv(ocsec.ENV_ALLOW_UNSAFE_DESERIALIZATION, "1")

    # Should not raise when environment opt-in is set
    ocsec.require_trusted_input(
        trusted=False,
        operation="pickle deserialization",
        env_var=ocsec.ENV_ALLOW_UNSAFE_DESERIALIZATION,
        source="trusted.pkl",
    )


@pytest.mark.order(75)
def test_require_trusted_input_raises_with_guidance(monkeypatch):
    monkeypatch.delenv(ocsec.ENV_ALLOW_UNSAFE_DESERIALIZATION, raising=False)

    with pytest.raises(PermissionError) as exc:
        ocsec.require_trusted_input(
            trusted=False,
            operation="pickle deserialization",
            env_var=ocsec.ENV_ALLOW_UNSAFE_DESERIALIZATION,
            source="untrusted.pkl",
        )

    message = str(exc.value)
    assert "allow_unsafe_runtime()" in message
    assert ocsec.ENV_ALLOW_UNSAFE_DESERIALIZATION in message
    assert "untrusted.pkl" in message


@pytest.mark.order(76)
def test_allow_unsafe_runtime_sets_selected_flags(monkeypatch):
    monkeypatch.delenv(ocsec.ENV_ALLOW_UNSAFE_DESERIALIZATION, raising=False)
    monkeypatch.delenv(ocsec.ENV_ALLOW_SCRIPT_EXEC, raising=False)

    ocsec.allow_unsafe_runtime(deserialization=True, script_exec=False)
    assert ocsec.env_flag_enabled(ocsec.ENV_ALLOW_UNSAFE_DESERIALIZATION) is True
    assert ocsec.env_flag_enabled(ocsec.ENV_ALLOW_SCRIPT_EXEC) is False

    monkeypatch.delenv(ocsec.ENV_ALLOW_UNSAFE_DESERIALIZATION, raising=False)
    monkeypatch.delenv(ocsec.ENV_ALLOW_SCRIPT_EXEC, raising=False)

    ocsec.allow_unsafe_runtime(deserialization=False, script_exec=True)
    assert ocsec.env_flag_enabled(ocsec.ENV_ALLOW_UNSAFE_DESERIALIZATION) is False
    assert ocsec.env_flag_enabled(ocsec.ENV_ALLOW_SCRIPT_EXEC) is True

    # Prevent env leakage to subsequent tests
    monkeypatch.delenv(ocsec.ENV_ALLOW_UNSAFE_DESERIALIZATION, raising=False)
    monkeypatch.delenv(ocsec.ENV_ALLOW_SCRIPT_EXEC, raising=False)
