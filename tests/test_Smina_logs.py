import pytest
from pathlib import Path

import OCDocker.Docking.Smina as ocsmina


def make_log(path: Path, lines: str) -> Path:
    path.write_text(lines)
    return path

def test_read_log(tmp_path):
    log_file = tmp_path / "dock.log"
    # minimal smina log section
    lines = (
        "-----+------------+----------+----------+\n"
        "    1 -7.5 0 0\n"
        "    2 -6.5 0 0\n"
    )
    make_log(log_file, lines)

    data = ocsmina.read_log(str(log_file))
    assert data[1][ocsmina.smina_scoring] == "-7.5" # type: ignore
    assert data[2][ocsmina.smina_scoring] == "-6.5" # type: ignore

    best = ocsmina.read_log(str(log_file), onlyBest=True)
    assert list(best.keys()) == [1]
    assert best[1][ocsmina.smina_scoring] == "-7.5" # type: ignore

def test_rescoring_logs(tmp_path):
    f1 = tmp_path / "lig_split_1_vinardo_rescoring.log"
    f2 = tmp_path / "lig_split_2_vinardo_rescoring.log"
    make_log(f1, "Affinity:            -7.0 (kcal/mol)\n")
    make_log(f2, "Affinity:            -6.5 (kcal/mol)\n")

    paths = ocsmina.get_rescore_log_paths(str(tmp_path))
    assert set(paths) == {str(f1), str(f2)}

    val1 = ocsmina.read_rescoring_log(str(f1))
    assert val1 == -7.0

    data = ocsmina.read_rescore_logs(paths)
    expected = {
        "rescoring_vinardo_1": -7.0,
        "rescoring_vinardo_2": -6.5,
    }
    assert data == expected

    best = ocsmina.read_rescore_logs(paths, onlyBest=True)
    assert best == {"rescoring_vinardo_1": -7.0}
