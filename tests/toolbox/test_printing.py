#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Toolbox.Printing helpers.
'''

# Imports
###############################################################################
import datetime
import importlib
import logging
import pytest
import sys
import types

import OCDocker.Error as ocerror

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

def _set_level(level):
    prev = ocerror.Error.get_output_level()
    ocerror.Error.set_output_level(level)
    return prev


def _captured_text(capsys, caplog):
    captured = capsys.readouterr()
    return f"{captured.out}\n{captured.err}\n{caplog.text}"


## Public ##

@pytest.fixture
def ocprint(monkeypatch):
    # Provide a minimal OCDocker.Initialise module so Printing can be imported
    fake_init = types.ModuleType("OCDocker.Initialise")
    fake_init.clrs = {k: "" for k in ["r", "g", "y", "b", "p", "c", "n"]} # type: ignore
    fake_init.ocerror = ocerror # type: ignore
    monkeypatch.setitem(sys.modules, "OCDocker.Initialise", fake_init)
    ocprint = importlib.import_module("OCDocker.Toolbox.Printing")
    importlib.reload(ocprint)
    # Replace the datetime dependency with a small shim that supports both
    # datetime.now() and datetime.datetime.now() usages.
    class _DT:
        @staticmethod
        def now():
            return datetime.datetime.now()



    _DT.datetime = datetime.datetime # type: ignore
    monkeypatch.setattr(ocprint, "datetime", _DT)
    # Prevent the helper from writing progress files during tests
    from io import StringIO
    import builtins



    def fake_open(*args, **kwargs):
        return StringIO()



    monkeypatch.setattr(builtins, "open", fake_open)
    yield ocprint
    monkeypatch.delitem(sys.modules, "OCDocker.Toolbox.Printing", raising=False)
    monkeypatch.delitem(sys.modules, "OCDocker.Initialise", raising=False)


@pytest.mark.order(77)
def test_print_error_contains_tag(ocprint, capsys, caplog):
    prev = _set_level(ocerror.ReportLevel.ERROR)
    try:
        with caplog.at_level(logging.ERROR, logger="ocdocker"):
            ocprint.print_error("fail")
    finally:
        ocerror.Error.set_output_level(prev)
    txt = _captured_text(capsys, caplog)
    assert "ERROR" in txt
    assert "fail" in txt


@pytest.mark.order(74)
def test_print_info_contains_tag(ocprint, capsys, caplog):
    prev = _set_level(ocerror.ReportLevel.INFO)
    try:
        with caplog.at_level(logging.INFO, logger="ocdocker"):
            ocprint.print_info("info message")
    finally:
        ocerror.Error.set_output_level(prev)
    txt = _captured_text(capsys, caplog)
    assert "INFO" in txt
    assert "info message" in txt


@pytest.mark.order(78)
def test_print_section_outputs_header(ocprint, capsys, tmp_path):
    prev = _set_level(ocerror.ReportLevel.DEBUG)
    try:
        ocprint.print_section(1, "Test", logName=str(tmp_path / "prog.log"))
    finally:
        ocerror.Error.set_output_level(prev)
    captured = capsys.readouterr()
    assert "S|E|C|T|I|O|N" in captured.out
    assert "Test" in captured.out


@pytest.mark.order(82)
def test_print_sorry_outputs_message(ocprint, capsys):
    prev = _set_level(ocerror.ReportLevel.DEBUG)
    try:
        ocprint.print_sorry()
    finally:
        ocerror.Error.set_output_level(prev)
    captured = capsys.readouterr()
    assert "sorry" in captured.out.lower()


@pytest.mark.order(80)
def test_print_subsection_outputs_header(ocprint, capsys, tmp_path):
    prev = _set_level(ocerror.ReportLevel.DEBUG)
    try:
        ocprint.print_subsection(1, "Sub", logName=str(tmp_path / "prog.log"))
    finally:
        ocerror.Error.set_output_level(prev)
    captured = capsys.readouterr()
    assert "Subsect" in captured.out  # part of the word subsection with bars
    assert "Sub" in captured.out


@pytest.mark.order(75)
def test_print_success_contains_tag(ocprint, capsys, caplog):
    prev = _set_level(ocerror.ReportLevel.SUCCESS)
    try:
        with caplog.at_level(logging.INFO, logger="ocdocker"):
            ocprint.print_success("great")
    finally:
        ocerror.Error.set_output_level(prev)
    txt = _captured_text(capsys, caplog)
    assert "SUCCESS" in txt
    assert "great" in txt


@pytest.mark.order(76)
def test_print_warning_contains_tag(ocprint, capsys, caplog):
    prev = _set_level(ocerror.ReportLevel.WARNING)
    try:
        with caplog.at_level(logging.WARNING, logger="ocdocker"):
            ocprint.print_warning("caution")
    finally:
        ocerror.Error.set_output_level(prev)
    txt = _captured_text(capsys, caplog)
    assert "WARNING" in txt
    assert "caution" in txt


@pytest.mark.order(73)
def test_printv_outputs_message(ocprint, capsys, caplog):
    prev = _set_level(ocerror.ReportLevel.DEBUG)
    try:
        with caplog.at_level(logging.DEBUG, logger="ocdocker"):
            ocprint.printv("hello")
    finally:
        ocerror.Error.set_output_level(prev)
    txt = _captured_text(capsys, caplog)
    assert "hello" in txt


@pytest.mark.order(79)
def test_section_returns_string(ocprint):
    result = ocprint.section(2, "Sec")
    assert isinstance(result, str)
    assert "S|E|C|T|I|O|N" in result
    assert "Sec" in result


@pytest.mark.order(81)
def test_subsection_returns_string(ocprint):
    result = ocprint.subsection(3, "Sub")
    assert isinstance(result, str)
    assert "Subsect" in result
    assert "Sub" in result
