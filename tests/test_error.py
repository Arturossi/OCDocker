import pytest
import OCDocker.Error as ocerror


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
