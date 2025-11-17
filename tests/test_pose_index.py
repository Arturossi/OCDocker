import pytest

import OCDocker.Docking.Vina as ocvina
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.PLANTS as ocplants


@pytest.mark.parametrize(
    "func,file_name,expected",
    [
        (ocvina.get_pose_index_from_file_path, "pose_split_5.pdbqt", 5),
        (ocsmina.get_pose_index_from_file_path, "pose_split_5.pdbqt", 5),
        (ocplants.get_pose_index_from_file_path, "ligand_pose_3.mol2", 3),
    ],
)
@pytest.mark.order(72)
def test_get_pose_index_from_file_path(func, file_name, expected):
    assert func(file_name) == expected
