import json
import pytest
import tarfile

import numpy as np

import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Error as ocerror


@pytest.mark.order(1)
def test_untar_and_delete(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    test_file = src / "test.txt"
    test_file.write_text("hello")

    archive = tmp_path / "archive.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(test_file, arcname="test.txt")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    rc = ocff.untar(str(archive), str(out_dir))
    assert rc == ocerror.ErrorCode.OK
    assert (out_dir / "test.txt").exists()

    del_rc = ocff.safe_remove_file(str(archive))
    assert del_rc == ocerror.ErrorCode.OK
    assert not archive.exists()


@pytest.mark.order(2)
def test_empty_docking_digest_json(tmp_path):
    digest_file = tmp_path / "digest.json"
    digest = ocff.empty_docking_digest(str(digest_file), overwrite=True, digestFormat="json")

    expected_keys = {
        "smina_pose",
        "smina_affinity",
        "PLANTS_TOTAL_SCORE",
        "PLANTS_SCORE_RB_PEN",
        "PLANTS_SCORE_NORM_HEVATOMS",
        "PLANTS_SCORE_NORM_CRT_HEVATOMS",
        "PLANTS_SCORE_NORM_WEIGHT",
        "PLANTS_SCORE_NORM_CRT_WEIGHT",
        "PLANTS_SCORE_RB_PEN_NORM_CRT_HEVATOMS",
        "vina_pose",
        "vina_affinity",
    }

    assert isinstance(digest, dict)
    assert set(digest.keys()) == expected_keys
    assert digest_file.exists()
    with open(digest_file) as f:
        data = json.load(f)
    assert set(data.keys()) == expected_keys
    for v in data.values():
        assert np.isnan(v[0]) # type: ignore


@pytest.mark.order(3)
def test_hdf5_round_trip(tmp_path):
    data = {"a": np.array([1, 2, 3]), "b": np.array([4.0])} # type: ignore
    file_path = tmp_path / "data.h5"
    result = ocff.to_hdf5(str(file_path), data)
    assert result == ocerror.ErrorCode.OK

    loaded = ocff.from_hdf5(str(file_path))
    assert isinstance(loaded, dict)
    assert set(loaded.keys()) == {"a", "b"}
    np.testing.assert_array_equal(loaded["a"], data["a"]) # type: ignore
    np.testing.assert_array_equal(loaded["b"], data["b"]) # type: ignore


@pytest.mark.order(4)
def test_pickle_round_trip(tmp_path):
    obj = {"x": [1, 2], "y": "test"}
    pkl = tmp_path / "obj.pkl"
    code = ocff.to_pickle(str(pkl), obj)
    assert code == ocerror.ErrorCode.OK
    loaded = ocff.from_pickle(str(pkl))
    assert loaded == obj


@pytest.mark.order(5)
def test_safe_create_and_remove_dir(tmp_path):
    dir_path = tmp_path / "newdir"
    code_create = ocff.safe_create_dir(str(dir_path))
    assert code_create == ocerror.ErrorCode.OK
    code_exists = ocff.safe_create_dir(str(dir_path))
    assert code_exists == ocerror.ErrorCode.DIR_EXISTS

    code_remove = ocff.safe_remove_dir(str(dir_path))
    assert code_remove == ocerror.ErrorCode.OK
    code_remove_again = ocff.safe_remove_dir(str(dir_path))
    assert code_remove_again == ocerror.ErrorCode.DIR_NOT_EXIST


@pytest.mark.order(6)
def test_safe_remove_file(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("content")
    code_remove = ocff.safe_remove_file(str(file_path))
    assert code_remove == ocerror.ErrorCode.OK
    code_remove_again = ocff.safe_remove_file(str(file_path))
    assert code_remove_again == ocerror.ErrorCode.FILE_NOT_EXIST
