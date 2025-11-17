import pytest

import OCDocker.Toolbox.Running as ocrun
import OCDocker.Error as ocerror


@pytest.mark.order(1)
def test_run_empty_cmd():
    result = ocrun.run([])
    assert result == ocerror.ErrorCode.NOT_SET


@pytest.mark.order(2)
def test_run_wrong_type():
    result = ocrun.run('notalist') # type: ignore
    assert result == ocerror.ErrorCode.WRONG_TYPE


@pytest.mark.order(3)
def test_run_echo(tmp_path):
    log = tmp_path / "run.log"
    code = ocrun.run(['echo', 'hello'], logFile=str(log))
    assert code == ocerror.ErrorCode.OK
    assert log.exists()
