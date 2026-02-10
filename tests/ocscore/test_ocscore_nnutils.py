#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore NN analysis utility helpers.
'''

# Imports
###############################################################################
import numpy as np

import pytest

import OCDocker.OCScore.Analysis.NNUtils as ocnnutils
import OCDocker.OCScore.Analysis.Plotting.Core as ocplotcore

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

class _FakePermutationResult:
    importances_mean = np.array([0.4, 0.1], dtype=float)


class _FakeAEModel:
    def predict(self, X):
        return X[:, 0]


## Public ##

@pytest.mark.order(259)
def test_run_ae_feature_importance_requires_predict():
    with pytest.raises(ValueError, match="predict"):
        ocnnutils.run_ae_feature_importance(
            ae_model=object(),
            X_valid=np.array([[1.0, 2.0]], dtype=float),
            y_valid=np.array([1.0], dtype=float),
            features=["f1", "f2"],
            n_repeats=1,
            save_dir="unused",
            prefix="X",
        )


@pytest.mark.order(260)
def test_run_ae_feature_importance_happy_path(monkeypatch, tmp_path):
    saved = []

    def _fake_permutation_importance(estimator, X, y, scoring, n_repeats, random_state):
        _ = (estimator, y, n_repeats, random_state)
        score = scoring(X)
        assert isinstance(score, float)
        return _FakePermutationResult()

    monkeypatch.setattr(ocnnutils, "permutation_importance", _fake_permutation_importance)
    monkeypatch.setattr(ocnnutils.sns, "barplot", lambda **kwargs: kwargs)
    monkeypatch.setattr(ocnnutils.plt, "savefig", lambda path: saved.append(path))

    X_valid = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=float)
    y_valid = np.array([1.0, 2.0, 3.0], dtype=float)

    out = ocnnutils.run_ae_feature_importance(
        ae_model=_FakeAEModel(),
        X_valid=X_valid,
        y_valid=y_valid,
        features=["f0", "f1"],
        n_repeats=3,
        save_dir=str(tmp_path / "plots"),
        prefix="AE",
    )

    assert list(out.columns) == ["Feature", "Importance"]
    assert out.iloc[0]["Feature"] == "f0"
    assert saved
    assert saved[0].endswith("AE_permutation_importance.png")


@pytest.mark.order(261)
def test_plotting_core_style_and_new_figure():
    ocplotcore.apply_basic_style()
    fig, ax = ocplotcore.new_fig(size=(4.0, 3.0))

    assert fig.get_size_inches()[0] == pytest.approx(4.0)
    assert fig.get_size_inches()[1] == pytest.approx(3.0)
    assert ax is not None

    ocplotcore.plt.close(fig)
