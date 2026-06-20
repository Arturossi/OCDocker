#!/usr/bin/env python3

# Description
###############################################################################
'''
Targeted branch coverage tests for BaseVinaLike helpers.
'''

# Imports
###############################################################################
import builtins
import errno
import json

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import OCDocker.Docking.BaseVinaLike as ocbasevina
import OCDocker.Error as ocerror

# License
###############################################################################
'''OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##

def test_generate_digest_wrong_type_and_box_merge_and_write_error(monkeypatch, tmp_path):
    digest_path = tmp_path / "digest.json"
    digest_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(ocbasevina.ocvalidation, "validate_digest_extension", lambda *_a, **_k: True)

    # Existing file + non-json format keeps digest as None and should trigger wrong_type branch.
    rc_wrong_type = ocbasevina._generate_digest_generic(
        str(digest_path),
        "dummy.log",
        read_log_func=lambda _p: {1: {"vina_score": -7.0}},
        overwrite=True,
        digestFormat="txt",
    )
    assert rc_wrong_type == ocerror.Error.wrong_type()

    digest_path.write_text(json.dumps({"box1": "invalid"}), encoding="utf-8")

    rc_ok = ocbasevina._generate_digest_generic(
        str(digest_path),
        "dummy.log",
        read_log_func=lambda _p: {1: {"vina_score": -8.0}},
        overwrite=True,
        digestFormat="json",
        box_id="box1",
    )
    assert rc_ok == ocerror.Error.ok()

    merged = json.loads(digest_path.read_text(encoding="utf-8"))
    assert isinstance(merged["box1"], dict)
    assert "1" in merged["box1"] or 1 in merged["box1"]

    real_open = builtins.open

    def broken_open(file, mode="r", *args, **kwargs):
        if str(file) == str(digest_path) and "w" in mode:
            raise OSError("denied")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", broken_open)

    rc_write = ocbasevina._generate_digest_generic(
        str(digest_path),
        "dummy.log",
        read_log_func=lambda _p: {1: {"vina_score": -9.0}},
        overwrite=True,
        digestFormat="json",
    )
    assert rc_write == ocerror.Error.write_file()


def test_read_log_and_rescoring_ioerror_and_parse_paths(monkeypatch, tmp_path):
    log_file = tmp_path / "vina.log"
    log_file.write_text("content\n", encoding="utf-8")

    # Parse branch with an invalid float value that should be skipped.
    monkeypatch.setattr(
        ocbasevina.ocio,
        "lazyread_reverse_order_mmap",
        lambda _p: iter([
            "2 BAD 0 0",
            "1 -7.50 0 0",
            "-----+------------+----------+----------+",
        ]),
    )

    data = ocbasevina._read_log_generic(
        str(log_file),
        scoring_key="vina_score",
        engine="vina",
        error_log="err.log",
    )
    assert list(data.keys()) == [1]

    def raise_epipe(_p):
        raise IOError(errno.EPIPE, "broken pipe")

    monkeypatch.setattr(ocbasevina.ocio, "lazyread_reverse_order_mmap", raise_epipe)
    monkeypatch.setattr(ocbasevina, "get_config", lambda: SimpleNamespace(logdir=str(tmp_path)))
    monkeypatch.setattr(ocbasevina.ocprint, "print_error", lambda *_a, **_k: None)
    monkeypatch.setattr(ocbasevina.ocprint, "print_error_log", lambda *_a, **_k: None)

    epipe_data = ocbasevina._read_log_generic(
        str(log_file),
        scoring_key="vina_score",
        engine="vina",
        error_log="err.log",
    )
    assert epipe_data == {}

    resc_epipe = ocbasevina._read_rescoring_log_generic(
        str(log_file),
        start_string="Affinity",
        engine="smina",
        error_log="err.log",
    )
    assert np.isnan(resc_epipe)

    monkeypatch.setattr(
        ocbasevina.ocio,
        "lazyread_reverse_order_mmap",
        lambda _p: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    resc_exception = ocbasevina._read_rescoring_log_generic(
        str(log_file),
        start_string="Affinity",
        engine="smina",
        error_log="err.log",
    )
    assert np.isnan(resc_exception)
