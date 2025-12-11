import importlib
import sys
import types
import datetime
import pytest

import OCDocker.Error as ocerror


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


def _set_level(level):
    prev = ocerror.Error.get_output_level()
    ocerror.Error.set_output_level(level)
    return prev


@pytest.mark.order(73)
def test_printv_outputs_message(ocprint, capsys):
    prev = _set_level(ocerror.ReportLevel.DEBUG)
    try:
        ocprint.printv("hello")
    finally:
        ocerror.Error.set_output_level(prev)
    captured = capsys.readouterr()
    assert "hello" in captured.out


@pytest.mark.order(74)
def test_print_info_contains_tag(ocprint, capsys):
    prev = _set_level(ocerror.ReportLevel.INFO)
    try:
        ocprint.print_info("info message")
    finally:
        ocerror.Error.set_output_level(prev)
    captured = capsys.readouterr()
    assert "INFO" in captured.out
    assert "info message" in captured.out


@pytest.mark.order(75)
def test_print_success_contains_tag(ocprint, capsys):
    prev = _set_level(ocerror.ReportLevel.SUCCESS)
    try:
        ocprint.print_success("great")
    finally:
        ocerror.Error.set_output_level(prev)
    captured = capsys.readouterr()
    assert "SUCCESS" in captured.out
    assert "great" in captured.out


@pytest.mark.order(76)
def test_print_warning_contains_tag(ocprint, capsys):
    prev = _set_level(ocerror.ReportLevel.WARNING)
    try:
        ocprint.print_warning("caution")
    finally:
        ocerror.Error.set_output_level(prev)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "caution" in captured.out


@pytest.mark.order(77)
def test_print_error_contains_tag(ocprint, capsys):
    prev = _set_level(ocerror.ReportLevel.ERROR)
    try:
        ocprint.print_error("fail")
    finally:
        ocerror.Error.set_output_level(prev)
    captured = capsys.readouterr()
    assert "ERROR" in captured.out
    assert "fail" in captured.out


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


@pytest.mark.order(79)
def test_section_returns_string(ocprint):
    result = ocprint.section(2, "Sec")
    assert isinstance(result, str)
    assert "S|E|C|T|I|O|N" in result
    assert "Sec" in result


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


@pytest.mark.order(81)
def test_subsection_returns_string(ocprint):
    result = ocprint.subsection(3, "Sub")
    assert isinstance(result, str)
    assert "Subsect" in result
    assert "Sub" in result


@pytest.mark.order(82)
def test_print_sorry_outputs_message(ocprint, capsys):
    prev = _set_level(ocerror.ReportLevel.DEBUG)
    try:
        ocprint.print_sorry()
    finally:
        ocerror.Error.set_output_level(prev)
    captured = capsys.readouterr()
    assert "sorry" in captured.out.lower()
