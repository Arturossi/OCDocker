import pytest

import OCDocker.Error as ocerror
import OCDocker.Toolbox.Printing as ocprint


@pytest.mark.order(83)
def test_printv_gated_by_level(capsys):
    # Ensure DEBUG prints
    ocerror.Error.set_output_level(ocerror.ReportLevel.DEBUG)
    ocprint.printv("visible")
    out = capsys.readouterr().out
    assert "visible" in out

    # Ensure non-DEBUG suppresses
    ocerror.Error.set_output_level(ocerror.ReportLevel.INFO)
    ocprint.printv("hidden")
    out2 = capsys.readouterr().out
    assert "hidden" not in out2


@pytest.mark.order(84)
def test_print_to_log_files(tmp_path):
    log = tmp_path / "out.log"
    ocprint.print_info_log("alpha", str(log))
    ocprint.print_warning_log("beta", str(log))
    ocprint.print_error_log("gamma", str(log))
    txt = log.read_text()
    assert "INFO: alpha" in txt
    assert "WARNING: beta" in txt
    assert "ERROR: gamma" in txt
