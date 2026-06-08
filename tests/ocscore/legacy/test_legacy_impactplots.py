#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Analysis.Plotting.ImpactPlots helpers.
'''

# Imports
###############################################################################
import pandas as pd

import pytest

import OCDocker.OCScore.Analysis.legacy.ImpactPlots as ocimpactplots

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

def _contingency_int() -> pd.DataFrame:
    return pd.DataFrame(
        [[6, 4, 0], [2, 8, 2]],
        index=[0, 1],
        columns=["Low", "Mid", "High"],
    )


def _residuals_dict() -> dict[str, pd.DataFrame]:
    return {
        "f1": pd.DataFrame([[0.2, -0.1], [2.3, -2.1]], index=[0, 1], columns=["A", "B"]),
        "f2": pd.DataFrame([[0.5, 0.4], [1.1, -0.3]], index=["absent", "present"], columns=["A", "B"]),
    }


## Public ##

@pytest.mark.order(642)
def test_prop_delta_2xk_supports_index_modes_and_rejects_non_2xk():
    out_int = ocimpactplots.prop_delta_2xk(_contingency_int())
    assert list(out_int.columns) == ["MetricCategory", "prop_delta"]
    assert set(out_int["MetricCategory"]) == {"Low", "Mid", "High"}

    cont_str = pd.DataFrame([[9, 1], [2, 8]], index=["0", "1"], columns=["Low", "High"])
    out_str = ocimpactplots.prop_delta_2xk(cont_str)
    assert len(out_str) == 2

    cont_fallback = pd.DataFrame([[1, 9], [8, 2]], index=["absence", "presence"], columns=["Low", "High"])
    out_fallback = ocimpactplots.prop_delta_2xk(cont_fallback)
    assert out_fallback.loc[out_fallback["MetricCategory"] == "Low", "prop_delta"].iloc[0] > 0

    with pytest.raises(ValueError, match="2xK"):
        ocimpactplots.prop_delta_2xk(
            pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=[0, 1, 2], columns=["a", "b"])
        )


@pytest.mark.order(643)
def test_residuals_matrix_from_dict_selects_presence_or_fallback():
    mat = ocimpactplots.residuals_matrix_from_dict(_residuals_dict(), presence_level=1)
    assert list(mat.columns) == ["A", "B"]
    assert "f1" in mat.index
    assert "f2" in mat.index
    assert mat.loc["f1", "A"] == pytest.approx(2.3)
    assert mat.loc["f2", "A"] == pytest.approx(1.1)


@pytest.mark.order(644)
def test_plot_helpers_save_outputs(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(ocimpactplots.plt, "savefig", lambda path, **_k: saved.append(path))
    monkeypatch.setattr(ocimpactplots.plt, "tight_layout", lambda *_a, **_k: None)
    monkeypatch.setattr(ocimpactplots.plt, "text", lambda *_a, **_k: None)
    monkeypatch.setattr(
        ocimpactplots.sns,
        "barplot",
        lambda *_a, **_k: _k.get("ax", ocimpactplots.plt.gca()),
    )

    contingency = _contingency_int()
    residuals = pd.DataFrame(
        [[-0.5, 0.3, 0.1], [2.1, -1.3, 0.6]],
        index=[0, 1],
        columns=["Low", "Mid", "High"],
    )
    residuals_dict = _residuals_dict()

    ocimpactplots.plot_prop_delta(contingency, outpath=str(tmp_path / "prop.png"))
    ocimpactplots.plot_chi2_contrib(contingency, feature_name="f1", presence_level=5, outpath=str(tmp_path / "chi2.png"))
    ocimpactplots.plot_residuals_lollipop(
        residuals,
        feature_name="f1",
        presence_level="missing",
        outpath=str(tmp_path / "lollipop.png"),
    )
    ocimpactplots.plot_residuals_matrix(
        residuals_dict,
        presence_level="missing",
        order_by="chi2",
        outpath=str(tmp_path / "matrix.png"),
    )

    assert any(str(p).endswith("prop.png") for p in saved)
    assert any(str(p).endswith("chi2.png") for p in saved)
    assert any(str(p).endswith("lollipop.png") for p in saved)
    assert any(str(p).endswith("matrix.png") for p in saved)


@pytest.mark.order(645)
def test_feature_report_2xk_saves_and_supports_optional_p_value(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(ocimpactplots.plt, "savefig", lambda path, **_k: saved.append(path))
    monkeypatch.setattr(ocimpactplots.plt, "tight_layout", lambda *_a, **_k: None)
    monkeypatch.setattr(
        ocimpactplots.sns,
        "barplot",
        lambda *_a, **_k: _k.get("ax", ocimpactplots.plt.gca()),
    )

    contingency = _contingency_int()
    residuals = pd.DataFrame(
        [[-0.5, 0.3, 0.1], [2.1, -1.3, 0.6]],
        index=["0", "1"],
        columns=["Low", "Mid", "High"],
    )

    ocimpactplots.feature_report_2xk(
        feature="f1",
        contingency=contingency,
        residuals_df=residuals,
        p_value=0.0123,
        outpath=str(tmp_path / "feature_report.png"),
    )

    assert len(saved) == 1
    assert str(saved[0]).endswith("feature_report.png")


@pytest.mark.order(646)
def test_plot_helpers_skip_save_when_no_outpath(monkeypatch):
    saved = []
    monkeypatch.setattr(ocimpactplots.plt, "savefig", lambda *_a, **_k: saved.append("called"))
    monkeypatch.setattr(ocimpactplots.plt, "tight_layout", lambda *_a, **_k: None)
    monkeypatch.setattr(ocimpactplots.plt, "text", lambda *_a, **_k: None)
    monkeypatch.setattr(
        ocimpactplots.sns,
        "barplot",
        lambda *_a, **_k: _k.get("ax", ocimpactplots.plt.gca()),
    )

    contingency = _contingency_int()
    residuals = pd.DataFrame([[0.1, 0.2], [0.3, -0.4]], index=[0, 1], columns=["A", "B"])

    ocimpactplots.plot_prop_delta(contingency, outpath=None)
    ocimpactplots.plot_chi2_contrib(contingency, feature_name="f2", outpath=None)
    ocimpactplots.plot_residuals_lollipop(residuals, feature_name="f2", outpath=None)

    assert saved == []
