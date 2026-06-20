#!/usr/bin/env python3

# Description
###############################################################################
'''
Extra tests for Toolbox.Validation helpers.
'''

# Imports
###############################################################################
import builtins

import pytest

import OCDocker.Toolbox.Validation as ocvalidation

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##

@pytest.mark.order(95)
def test_validate_obabel_extension_and_digest_format(tmp_path):
    # Supported ext
    assert ocvalidation.validate_obabel_extension("foo.mol2") == "mol2"
    # Unsupported ext returns an int error code
    bad = ocvalidation.validate_obabel_extension("foo.zzz")
    assert isinstance(bad, int) and bad != 0

    # Digest format validation
    assert ocvalidation.validate_digest_extension(str(tmp_path/"x.json"), "json") is True
    # Unknown format: tries to infer from path
    assert ocvalidation.validate_digest_extension(str(tmp_path/"x.json"), "foobar") is True


@pytest.mark.order(96)
def test_validate_digest_extension_falls_back_to_print_error_when_warning_missing(monkeypatch):
    messages = []
    monkeypatch.delattr(ocvalidation.ocprint, "print_warning", raising=False)
    monkeypatch.setattr(ocvalidation.ocprint, "print_error", lambda msg: messages.append(msg), raising=False)

    assert ocvalidation.validate_digest_extension("digest.json", "unknownfmt") is True
    assert any(msg.startswith("WARNING:") for msg in messages)


@pytest.mark.order(97)
def test_validate_digest_extension_uses_builtin_print_when_printing_api_missing(monkeypatch):
    printed = []
    monkeypatch.delattr(ocvalidation.ocprint, "print_warning", raising=False)
    monkeypatch.delattr(ocvalidation.ocprint, "print_error", raising=False)
    monkeypatch.setattr(builtins, "print", lambda msg: printed.append(msg))

    assert ocvalidation.validate_digest_extension("digest.hdf5", "badfmt") is False
    assert any(str(msg).startswith("WARNING:") for msg in printed)
    assert any(str(msg).startswith("ERROR:") for msg in printed)
