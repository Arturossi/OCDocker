import os
from pathlib import Path

import pytest

import OCDocker.CLI.__init__ as cli


@pytest.mark.order(25)
def test_preparse_global_args_and_require(tmp_path):
    ns = cli._preparse_global_args([
        "vs", "--output-level", "5", "--conf", "cfg.ini", "--overwrite", "--no-stdout-log",
    ])
    assert ns.output_level == 5
    assert ns.config_file == "cfg.ini"
    assert ns.overwrite is True
    assert ns.no_stdout_log is True

    p = tmp_path / "x.pdb"
    p.write_text("ATOM\n")
    got = cli._require_file(str(p), "--receptor")
    assert isinstance(got, Path) and got.exists()

    with pytest.raises(SystemExit):
        _ = cli._require_file("…/fake.pdb", "--receptor")

