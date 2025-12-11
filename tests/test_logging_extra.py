import os

import pytest

import OCDocker.Error as ocerror
import OCDocker.Toolbox.Logging as oclogging


@pytest.mark.order(69)
def test_logging_config_and_file(tmp_path):
    log_file = tmp_path / "test.log"

    # Configure with file handler and info level
    oclogging.configure(level=ocerror.Error.get_output_level(), log_file=str(log_file))
    logger = oclogging.get_logger("unit")
    logger.info("hello world")

    assert log_file.exists()
    contents = log_file.read_text()
    assert "hello world" in contents


@pytest.mark.order(70)
def test_backup_log(tmp_path):
    # Prepare a file in the default logdir matching backup_log expectation
    logdir = oclogging._default_logdir()
    os.makedirs(logdir, exist_ok=True)
    src = os.path.join(logdir, "unit_test.log")
    with open(src, "w") as f:
        f.write("x")

    # Backup and check it's moved under read_log_past
    oclogging.backup_log("unit_test")
    assert not os.path.exists(src)
    past_dir = os.path.join(logdir, "read_log_past")
    assert os.path.isdir(past_dir)
    past_files = [p for p in os.listdir(past_dir) if p.startswith("unit_test_")]
    assert past_files, "backup file not created"

