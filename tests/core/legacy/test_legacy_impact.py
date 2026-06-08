#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Analysis.Impact helpers.
'''

# Imports
###############################################################################
import numpy as np
import pandas as pd

import pytest

import OCDocker.OCScore.Analysis.legacy.Impact as ocimpact

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

def _chi_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Feature": "f1", "Cramér's V": 0.35, "Chi2 Statistic": 12.1, "p-value": 1e-4},
            {"Feature": "f2", "Cramér's V": 0.05, "Chi2 Statistic": 0.8, "p-value": 0.5},
            {"Feature": "f_missing", "Cramér's V": np.nan, "Chi2 Statistic": np.nan, "p-value": np.nan},
        ]
    )


def _contingency() -> dict[str, pd.DataFrame]:
    return {
        "f1": pd.DataFrame(
            [[8, 2], [3, 7]],
            index=[0, 1],
            columns=["Low", "High"],
        ),
        "f2": pd.DataFrame(
            [[5, 5], [6, 4]],
            index=["0", "1"],
            columns=["Low", "High"],
        ),
    }


## Public ##

@pytest.mark.order(741)
def test_impact_category_and_strength_helpers():
    assert ocimpact._beneficial_categories("AUC", ["Low", "High"]) == {"High"}
    assert ocimpact._beneficial_categories("RMSE", ["Low", "High"]) == {"Low"}
    assert ocimpact._beneficial_categories("OTHER", ["a", "b", "c", "d"]) == {"c", "d"}
    assert ocimpact._beneficial_categories("AUC", ["x", "y"], custom=["x"]) == {"x"}

    assert ocimpact._strength_from_nbs_norm(0.0) == "none"
    assert ocimpact._strength_from_nbs_norm(0.05) == "weak"
    assert ocimpact._strength_from_nbs_norm(0.15) == "moderate"
    assert ocimpact._strength_from_nbs_norm(0.25) == "strong"
    assert ocimpact._strength_from_nbs_norm(0.50) == "very strong"

    assert ocimpact._strength_from_v(np.nan) == "unknown"
    assert ocimpact._strength_from_v(0.05) == "none"
    assert ocimpact._strength_from_v(0.15) == "weak"
    assert ocimpact._strength_from_v(0.25) == "moderate"
    assert ocimpact._strength_from_v(0.40) == "strong"
    assert ocimpact._strength_from_v(0.55) == "very strong"


@pytest.mark.order(742)
def test_proportion_delta_and_net_benefit_fallback_keys():
    cont = pd.DataFrame([[1, 9], [8, 2]], index=["absent", "present"], columns=["Low", "High"])
    delta = ocimpact._proportion_delta(cont, presence_level=1)
    assert set(delta.index) == {"Low", "High"}
    assert delta["Low"] > 0
    assert delta["High"] < 0

    nbs = ocimpact._net_benefit(delta, beneficial={"High"})
    assert isinstance(nbs, float)
    assert nbs < 0


@pytest.mark.order(743)
def test_build_impact_overview_and_neutral_feature_selection():
    out = ocimpact.build_impact_overview(
        chi_df=_chi_df(),
        contingency_dict=_contingency(),
        metric="AUC",
        presence_level=1,
        tau=0.10,
    )
    assert {"Feature", "NBS", "Direction", "Strength", "NegLog10P"}.issubset(out.columns)
    assert "f_missing" in out["Feature"].values

    neutrals = ocimpact.get_neutral_features(out, tau=0.10)
    assert isinstance(neutrals, list)
    assert "f_missing" in neutrals

    out_no_dir = out.drop(columns=["Direction"])
    neutrals_by_nbs = ocimpact.get_neutral_features(out_no_dir, tau=0.10)
    assert isinstance(neutrals_by_nbs, list)


@pytest.mark.order(744)
def test_build_impact_overview_with_custom_beneficial_categories_changes_direction():
    out = ocimpact.build_impact_overview(
        chi_df=_chi_df().iloc[:2],
        contingency_dict=_contingency(),
        metric="AUC",
        beneficial_custom=["Low"],
        tau=0.01,
    )
    row_f1 = out.loc[out["Feature"] == "f1"].iloc[0]
    assert row_f1["Direction"] in {"positive", "negative", "neutral"}
    assert pd.notna(row_f1["FavoredCategory"])


@pytest.mark.order(745)
def test_plot_impact_arrows_inline_labels_saves_when_outpath_provided(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(ocimpact.plt, "savefig", lambda path, **_k: saved.append(path))

    impact_df = pd.DataFrame(
        {
            "Feature": ["f1", "f2", "f3"],
            "NBS": [0.4, -0.2, 0.01],
        }
    )
    out_png = tmp_path / "impact.png"
    ocimpact.plot_impact_arrows_inline_labels(
        impact_df=impact_df,
        title="Impact",
        outpath=str(out_png),
        tau=0.05,
    )

    assert saved
    assert saved[0].endswith("impact.png")
