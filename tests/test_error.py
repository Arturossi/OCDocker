#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Error reporting helpers and output levels.
'''

# Imports
###############################################################################
import pytest

import OCDocker.Error as ocerror

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

@pytest.mark.order(2)
def test_dynamic_error_methods_return_codes():
    assert ocerror.Error.file_not_exist() == ocerror.ErrorCode.FILE_NOT_EXIST # type: ignore
    assert ocerror.Error.dir_not_exist() == ocerror.ErrorCode.DIR_NOT_EXIST # type: ignore
    assert ocerror.Error.ok() == ocerror.ErrorCode.OK # type: ignore


@pytest.mark.order(3)
def test_print_message_outputs_formatted_string(capsys):
    prev = ocerror.Error.get_output_level()
    try:
        ocerror.Error.set_output_level(ocerror.ReportLevel.INFO)
        ocerror.Error.print_message("test message", ocerror.ReportLevel.INFO)
    finally:
        ocerror.Error.set_output_level(prev)

    captured = capsys.readouterr()
    assert "INFO" in captured.out
    assert "test message" in captured.out


@pytest.mark.order(1)
def test_set_output_level_enum_and_int():
    prev = ocerror.Error.get_output_level()
    try:
        ocerror.Error.set_output_level(ocerror.ReportLevel.WARNING)
        assert ocerror.Error.get_output_level() == ocerror.ReportLevel.WARNING

        ocerror.Error.set_output_level(3)
        assert ocerror.Error.get_output_level() == ocerror.ReportLevel.INFO
    finally:
        ocerror.Error.set_output_level(prev)


@pytest.mark.order(4)
def test_dynamic_error_methods_use_default_levels(capsys):
    prev = ocerror.Error.get_output_level()
    try:
        ocerror.Error.set_output_level(ocerror.ReportLevel.DEBUG)
        _ = ocerror.Error.file_not_exist("missing file test") # type: ignore
        _ = ocerror.Error.skip("skip test") # type: ignore
    finally:
        ocerror.Error.set_output_level(prev)

    captured = capsys.readouterr()
    assert "ERROR" in captured.out
    assert "missing file test" in captured.out
    assert "INFO" in captured.out
    assert "skip test" in captured.out


@pytest.mark.order(5)
def test_output_level_none_suppresses_messages(capsys):
    prev = ocerror.Error.get_output_level()
    try:
        ocerror.Error.set_output_level(ocerror.ReportLevel.NONE)
        _ = ocerror.Error.file_not_exist("this should not be shown") # type: ignore
        ocerror.Error.print_message("this should also not be shown", ocerror.ReportLevel.ERROR)
    finally:
        ocerror.Error.set_output_level(prev)

    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.order(6)
def test_print_message_respects_output_threshold(capsys):
    prev = ocerror.Error.get_output_level()
    try:
        ocerror.Error.set_output_level(ocerror.ReportLevel.ERROR)
        ocerror.Error.print_message("info should be hidden", ocerror.ReportLevel.INFO)
        ocerror.Error.print_message("error should be visible", ocerror.ReportLevel.ERROR)
    finally:
        ocerror.Error.set_output_level(prev)

    captured = capsys.readouterr()
    assert "info should be hidden" not in captured.out
    assert "ERROR" in captured.out
    assert "error should be visible" in captured.out
