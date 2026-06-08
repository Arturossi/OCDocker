#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for exported-model CLI data loading helpers.
'''

# Imports
###############################################################################
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import OCDocker.OCScore.CLI.export_tools as ocexportcli
import OCDocker.OCScore.Optimization.ModelExport as ocexport

# License
###############################################################################
'''
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
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##


def _export_dir(tmp_path, task: str):
    '''Create a minimal export directory for tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    task : str
        Export task name.

    Returns
    -------
    pathlib.Path
        Export directory path.
    '''

    export_dir = tmp_path / f"{task}_export"
    export_dir.mkdir()
    (export_dir / ocexport.RETRAIN_CONFIG_FILENAME).write_text(
        json.dumps({"task": task}),
        encoding="utf-8",
    )
    return export_dir


def _csv(tmp_path, name: str, marker: str):
    '''Create a small marker CSV file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    name : str
        CSV filename.
    marker : str
        Marker value written to the CSV.

    Returns
    -------
    pathlib.Path
        CSV path.
    '''

    path = tmp_path / name
    pd.DataFrame({"marker": [marker], "value": [1]}).to_csv(path, index=False)
    return path


def _args(*, pdbbind_csv=None, dudez_csv=None, reduction_archive=None):
    '''Create an argparse-like namespace for data-loading tests.

    Parameters
    ----------
    pdbbind_csv : pathlib.Path, optional
        PDBbind CSV path.
    dudez_csv : pathlib.Path, optional
        DUDEz CSV path.
    reduction_archive : pathlib.Path, optional
        Reduction archive path.

    Returns
    -------
    types.SimpleNamespace
        Namespace matching CLI argument attributes.
    '''

    return SimpleNamespace(
        pdbbind_csv=str(pdbbind_csv) if pdbbind_csv is not None else None,
        dudez_csv=str(dudez_csv) if dudez_csv is not None else None,
        reduction_archive=(
            str(reduction_archive) if reduction_archive is not None else None
        ),
    )


## Public ##


@pytest.mark.order(440)
def test_load_dataframe_for_dudez_export_prefers_dudez_csv_when_both_are_passed(
    tmp_path,
):
    '''Test DUDEz exports prefer the DUDEz CSV when both task CSVs are passed.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    '''

    pdbbind_csv = _csv(tmp_path, "pdbbind.csv", "pdbbind")
    dudez_csv = _csv(tmp_path, "dudez.csv", "dudez")
    export_dir = _export_dir(tmp_path, "dudez_screening")

    dataframe = ocexportcli._load_dataframe_for_export(
        export_dir,
        _args(pdbbind_csv=pdbbind_csv, dudez_csv=dudez_csv),
    )

    assert dataframe["marker"].tolist() == ["dudez"]


@pytest.mark.order(441)
def test_load_dataframe_for_pdbbind_export_prefers_pdbbind_csv_when_both_are_passed(
    tmp_path,
):
    '''Test PDBbind exports prefer the PDBbind CSV when both task CSVs are passed.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    '''

    pdbbind_csv = _csv(tmp_path, "pdbbind.csv", "pdbbind")
    dudez_csv = _csv(tmp_path, "dudez.csv", "dudez")
    export_dir = _export_dir(tmp_path, "pdbbind_regression")

    dataframe = ocexportcli._load_dataframe_for_export(
        export_dir,
        _args(pdbbind_csv=pdbbind_csv, dudez_csv=dudez_csv),
    )

    assert dataframe["marker"].tolist() == ["pdbbind"]


@pytest.mark.order(442)
def test_cmd_load_prints_selected_feature_summary(monkeypatch, tmp_path, capsys):
    '''Test compact selected-feature output for the load command.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Monkeypatch fixture.
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    capsys : pytest.CaptureFixture
        Output capture fixture.
    '''

    features = [f"feature_{idx}" for idx in range(20)]

    monkeypatch.setattr(
        ocexportcli.ocexport,
        "load_exported_model",
        lambda *_args, **_kwargs: {
            "export_dir": str(tmp_path / "best_model"),
            "summary": {
                "task": "pdbbind_regression",
                "trial_number": 7,
                "objective_metric": "RMSE",
                "final_objective_value": 1.2,
            },
            "selected_features": features,
            "model": object(),
        },
    )

    ocexportcli._cmd_load(
        SimpleNamespace(
            export_dir=str(tmp_path / "best_model"), retrain_from=None, device="cpu"
        )
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert "selected_features" not in payload
    assert payload["selected_features_count"] == 20
    assert payload["selected_features_preview"] == features[:8]
    assert payload["selected_features_omitted"] == 12
    assert "feature_19" not in output


@pytest.mark.order(443)
def test_cmd_retrain_prints_selected_feature_summary(monkeypatch, tmp_path, capsys):
    '''Test compact selected-feature output for the retrain command.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Monkeypatch fixture.
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    capsys : pytest.CaptureFixture
        Output capture fixture.
    '''

    features = [f"feature_{idx}" for idx in range(18)]

    monkeypatch.setattr(
        ocexportcli.ocexport,
        "retrain_from_export",
        lambda *_args, **_kwargs: {
            "retrain_config": {"task": "dudez_screening"},
            "selected_features": features,
            "splits": {
                "X_train": np.zeros((10, len(features))),
                "X_val": np.zeros((4, len(features))),
                "X_test": np.zeros((5, len(features))),
            },
            "model": object(),
        },
    )

    ocexportcli._cmd_retrain(
        SimpleNamespace(
            export_dir=str(tmp_path / "best_model"),
            retrain_from=None,
            pdbbind_csv=None,
            dudez_csv=None,
            device="cpu",
            ignore_saved_splits=False,
        )
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert "selected_features" not in payload
    assert payload["selected_features_count"] == 18
    assert payload["selected_features_preview"] == features[:8]
    assert payload["selected_features_omitted"] == 10
    assert payload["train_shape"] == [10, 18]
    assert "feature_17" not in output


@pytest.mark.order(444)
def test_cmd_cross_validate_prints_compact_scorer_summary(
    monkeypatch, tmp_path, capsys
):
    '''Test compact scorer comparison output for cross-validation.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Monkeypatch fixture.
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    capsys : pytest.CaptureFixture
        Output capture fixture.
    '''

    dataframe = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    result = SimpleNamespace(
        export_dir=str(tmp_path / "best_model"),
        task="dudez_screening",
        strategy="row_kfold",
        n_folds=5,
        effective_folds=5,
        epochs=2,
        objective_metric="BEDROC",
        aggregate_validation_metrics={"BEDROC": {"mean": 0.5, "std": 0.1}},
        scoring_function_columns=[f"sf_{idx}" for idx in range(30)],
        scorer_comparison_summary={
            "ocscore_wins": [
                {"metric": "BEDROC", "n_folds_won": 3, "n_folds_compared": 5},
                {"metric": "MCC", "n_folds_won": 0, "n_folds_compared": 0},
            ]
        },
    )

    monkeypatch.setattr(
        ocexportcli, "_load_dataframe_for_export", lambda *_args, **_kwargs: dataframe
    )
    monkeypatch.setattr(
        ocexportcli.occv,
        "run_cross_validation_from_export",
        lambda *_args, **_kwargs: result,
    )

    ocexportcli._cmd_cross_validate(
        SimpleNamespace(
            export_dir=str(tmp_path / "best_model"),
            retrain_from=None,
            reduction_archive=None,
            pdbbind_csv=None,
            dudez_csv=None,
            n_folds=5,
            epochs=2,
            seed=42,
            no_shuffle=False,
            strategy="auto",
            group_column="receptor",
            kind_column="kind",
            no_scoring_baselines=False,
            no_descriptor_aggregates=False,
            no_sf_consensus=False,
            output_dir=str(tmp_path / "cv"),
            device="cpu",
        )
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert "scoring_function_columns" not in payload
    assert "aggregate_scoring_function_metrics" not in payload
    assert payload["scoring_function_baselines_count"] == 30
    assert payload["ocscore_wins_preview"] == [
        {"metric": "BEDROC", "n_folds_won": 3, "n_folds_compared": 5}
    ]
    assert "sf_29" not in output


@pytest.mark.order(445)
def test_prepare_raw_dataframe_for_dudez_export_uses_ocscoreio(monkeypatch, tmp_path):
    '''Test raw DUDEz scoring data preparation delegates to IO helpers.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Monkeypatch fixture.
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    '''

    export_dir = _export_dir(tmp_path, "dudez_screening")
    raw = pd.DataFrame({"kind": ["ligand", "decoy"], "value": [1.0, 0.0]})
    prepared = raw.assign(dataset="dudez")
    calls = []

    def _load_pipeline_results(path, *, member_name=None):
        '''Record raw archive loading arguments.

        Parameters
        ----------
        path : str
            Raw archive path.
        member_name : str, optional
            Archive member name.

        Returns
        -------
        pandas.DataFrame
            Raw test dataframe.
        '''

        calls.append((path, member_name))
        return raw

    monkeypatch.setattr(
        ocexportcli.ocscoreio,
        "load_pipeline_results_from_archive",
        _load_pipeline_results,
    )
    monkeypatch.setattr(
        ocexportcli.ocscoreio, "prepare_dudez_dataframe", lambda frame: prepared
    )

    result = ocexportcli._prepare_raw_dataframe_for_export(
        export_dir,
        SimpleNamespace(
            raw_archive=str(tmp_path / "raw.tar"), archive_member="pipeline_results.csv"
        ),
    )

    assert calls == [(str(tmp_path / "raw.tar"), "pipeline_results.csv")]
    pd.testing.assert_frame_equal(result, prepared)


@pytest.mark.order(448)
def test_cmd_architecture_plot_prints_compact_summary(monkeypatch, tmp_path, capsys):
    '''Test compact architecture-plot command output.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Monkeypatch fixture.
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    capsys : pytest.CaptureFixture
        Output capture fixture.
    '''

    architecture_file = tmp_path / "architecture.json"
    architecture_file.write_text(
        json.dumps(
            {"task": "pdbbind_regression", "input_size": 4, "layers": [4, 2, 1]}
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_save_architecture_figures(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "png": str(tmp_path / "figures" / "architecture.png"),
            "svg": str(tmp_path / "figures" / "architecture.svg"),
            "source": str(architecture_file),
        }

    monkeypatch.setattr(
        ocexportcli.ocarchplot,
        "save_architecture_figures",
        fake_save_architecture_figures,
    )

    ocexportcli._cmd_architecture_plot(
        SimpleNamespace(
            architecture_file=str(architecture_file),
            export_dir=None,
            retrain_from=None,
            output_dir=str(tmp_path / "figures"),
            formats="png,svg",
            dpi=100,
            title=None,
            basename="architecture",
            show_decoder=True,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "architecture_figures_written"
    assert payload["task"] == "pdbbind_regression"
    assert payload["n_figures"] == 2
    assert "layers" not in payload
    assert sorted(payload["artifacts"]) == ["png", "svg"]
    assert calls[0][1]["include_decoder"] is True
