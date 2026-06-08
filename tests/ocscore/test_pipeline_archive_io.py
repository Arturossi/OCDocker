#!/usr/bin/env python3

# Description
###############################################################################
"""Tests for pipeline archive loading helpers."""

# Imports
###############################################################################
import io
import tarfile

import pandas as pd
import pytest

import OCDocker.OCScore.Utils.IO as ocscoreio

# License
###############################################################################
"""
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
"""


# Functions
###############################################################################
## Private ##

def _write_tar(path, members: dict[str, str]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members.items():
            payload = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


## Public ##

@pytest.mark.order(280)
def test_load_pipeline_results_from_tar(tmp_path):
    csv_text = "receptor,name,experimental\nr1,l1,1.0\n"
    tar_path = tmp_path / "archive.tar.gz"
    _write_tar(tar_path, {"pipeline_results.csv": csv_text})

    loaded = ocscoreio.load_pipeline_results_from_archive(tar_path)
    assert list(loaded.columns) == ["receptor", "name", "experimental"]
    assert len(loaded) == 1


@pytest.mark.order(281)
def test_load_pipeline_results_from_directory(tmp_path):
    csv_path = tmp_path / "pipeline_results.csv"
    csv_path.write_text("receptor,name,kind\nr1,l1,ligands\n", encoding="utf-8")

    loaded = ocscoreio.load_pipeline_results_from_archive(tmp_path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["kind"] == "ligands"


@pytest.mark.order(282)
def test_load_pipeline_results_empty_csv_raises(tmp_path):
    tar_path = tmp_path / "empty.tar.gz"
    _write_tar(tar_path, {"pipeline_results.csv": ""})

    with pytest.raises(ValueError, match="empty"):
        ocscoreio.load_pipeline_results_from_archive(tar_path)


@pytest.mark.order(283)
def test_load_pipeline_results_missing_csv_in_tar(tmp_path):
    tar_path = tmp_path / "missing.tar.gz"
    _write_tar(tar_path, {"other.csv": "a,b\n1,2\n"})

    with pytest.raises(FileNotFoundError, match="pipeline_results.csv"):
        ocscoreio.load_pipeline_results_from_archive(tar_path)


@pytest.mark.order(284)
def test_load_pipeline_results_multiple_members_requires_selection(tmp_path):
    tar_path = tmp_path / "multi.tar.gz"
    _write_tar(
        tar_path,
        {
            "a/pipeline_results.csv": "receptor,name\nr1,l1\n",
            "b/pipeline_results.csv": "receptor,name\nr2,l2\n",
        },
    )

    with pytest.raises(ValueError, match="member_name"):
        ocscoreio.load_pipeline_results_from_archive(tar_path)

    loaded = ocscoreio.load_pipeline_results_from_archive(
        tar_path,
        member_name="b/pipeline_results.csv",
    )
    assert loaded.iloc[0]["receptor"] == "r2"


@pytest.mark.order(285)
def test_directory_and_tar_load_identical_frames(tmp_path):
    csv_text = "receptor,name,experimental,f0\nr1,l1,1.0,0.5\n"
    directory = tmp_path / "dir_archive"
    directory.mkdir()
    (directory / "pipeline_results.csv").write_text(csv_text, encoding="utf-8")

    tar_path = tmp_path / "tar_archive.tar.gz"
    _write_tar(tar_path, {"pipeline_results.csv": csv_text})

    from_dir = ocscoreio.load_pipeline_results_from_archive(directory)
    from_tar = ocscoreio.load_pipeline_results_from_archive(tar_path)
    pd.testing.assert_frame_equal(from_dir, from_tar)


@pytest.mark.order(286)
def test_load_bare_pdbbind_csv_file(tmp_path):
    csv_path = tmp_path / "PDBbind.csv"
    csv_path.write_text("receptor,name,experimental\nr1,l1,1.0\n", encoding="utf-8")
    loaded = ocscoreio.load_pipeline_results_from_archive(csv_path)
    assert list(loaded.columns) == ["receptor", "name", "experimental"]
    assert len(loaded) == 1


@pytest.mark.order(287)
def test_load_directory_with_pdbbind_csv_only(tmp_path):
    csv_path = tmp_path / "PDBbind.csv"
    csv_path.write_text("receptor,name,experimental\nr1,l1,1.0\n", encoding="utf-8")
    loaded = ocscoreio.load_pipeline_results_from_archive(tmp_path)
    assert len(loaded) == 1


@pytest.mark.order(288)
def test_load_tar_with_dudez_csv_only(tmp_path):
    csv_text = "receptor,name,kind\nr1,l1,ligands\n"
    tar_path = tmp_path / "dudez.tar.gz"
    _write_tar(tar_path, {"DUDEz.csv": csv_text})
    loaded = ocscoreio.load_pipeline_results_from_archive(tar_path)
    assert loaded.iloc[0]["kind"] == "ligands"


@pytest.mark.order(289)
def test_load_directory_without_canonical_csv_raises(tmp_path):
    (tmp_path / "other.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Expected one of"):
        ocscoreio.load_pipeline_results_from_archive(tmp_path)


@pytest.mark.order(290)
def test_load_tar_with_two_canonical_csvs_requires_member_name(tmp_path):
    tar_path = tmp_path / "multi.tar.gz"
    _write_tar(
        tar_path,
        {
            "PDBbind.csv": "receptor,name,experimental\nr1,l1,1.0\n",
            "DUDEz.csv": "receptor,name,kind\nr2,l2,ligands\n",
        },
    )
    with pytest.raises(ValueError, match="member_name"):
        ocscoreio.load_pipeline_results_from_archive(tar_path)


@pytest.mark.order(291)
def test_prepare_pdbbind_dataframe_adds_dataset_column():
    raw = pd.DataFrame({"receptor": ["r1"], "experimental": [1.0]})
    prepared = ocscoreio.prepare_pdbbind_dataframe(raw)
    assert prepared.iloc[0]["dataset"] == "pdbbind"
    assert pd.isna(prepared.iloc[0]["label"])


@pytest.mark.order(287)
def test_prepare_dudez_dataframe_requires_kind():
    raw = pd.DataFrame({"receptor": ["r1"]})
    with pytest.raises(ValueError, match="kind"):
        ocscoreio.prepare_dudez_dataframe(raw)
