import pytest

import OCDocker.Docking.Vina as ocvina
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Docking.Smina as ocsmina


@pytest.mark.parametrize(
    "getter",
    [
        ocvina.get_docked_poses,
        ocplants.get_docked_poses,
        ocsmina.get_docked_poses,
    ],
)
@pytest.mark.order(66)
def test_get_docked_poses_missing_dir(getter, capsys, tmp_path):
    missing_dir = tmp_path / "non_existent"
    poses = getter(str(missing_dir))
    captured = capsys.readouterr()
    assert poses == []
    assert "does not exist" in captured.out
