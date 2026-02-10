#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Scoring.get_score.
'''

# Imports
###############################################################################
import numpy as np
import pandas as pd
import pytest
import sys
import torch
import types

import OCDocker.OCScore.Scoring as ocscoring

from types import SimpleNamespace

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


class _PredictModel:
    def predict(self, x):
        x_arr = np.asarray(x, dtype=float)
        return x_arr.sum(axis=1)


class _ForwardModel:
    def __init__(self):
        self.mask = {"mask": [1, 1]}

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def forward(self, x):
        return x.sum(dim=1, keepdim=True)

    __call__ = forward


class _CapturePredictModel:
    def __init__(self):
        self.seen_shape = None

    def predict(self, x):
        arr = np.asarray(x, dtype=float)
        self.seen_shape = arr.shape
        return arr.sum(axis=1)


class _ForwardFallbackMaskModel:
    def __init__(self):
        self.mask = {"unexpected": 123}

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def forward(self, x):
        return x.sum(dim=1, keepdim=True)

    __call__ = forward


# Functions
###############################################################################
## Private ##


def _model_file(tmp_path):
    p = tmp_path / "model.bin"
    p.write_text("model", encoding="utf-8")
    return p


def _scaler_file(tmp_path):
    p = tmp_path / "scaler.bin"
    p.write_text("scaler", encoding="utf-8")
    return p


def _base_df():
    return pd.DataFrame(
        {
            "receptor": ["rec1", "rec2"],
            "ligand": ["lig1", "lig2"],
            "feat1": [1.0, 2.0],
            "feat2": [3.0, 4.0],
        }
    )


## Public ##


@pytest.mark.order(371)
def test_get_score_raises_when_model_file_does_not_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        ocscoring.get_score(
            model_path=str(tmp_path / "missing-model.bin"),
            data=_base_df(),
            normalize=False,
            invert_conditionally=False,
            enforce_reference_column_order=False,
            use_gpu=False,
        )


@pytest.mark.order(372)
def test_get_score_raises_for_state_dict_payload(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: {"state_dict": {"x": 1}})

    with pytest.raises(ValueError, match="state_dict"):
        ocscoring.get_score(
            model_path=str(model_path),
            data=_base_df(),
            normalize=False,
            invert_conditionally=False,
            enforce_reference_column_order=False,
            use_gpu=False,
        )


@pytest.mark.order(373)
def test_get_score_raises_when_data_file_path_does_not_exist(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: _PredictModel())

    with pytest.raises(FileNotFoundError, match="Data file not found"):
        ocscoring.get_score(
            model_path=str(model_path),
            data=str(tmp_path / "missing-data.csv"),
            normalize=False,
            invert_conditionally=False,
            enforce_reference_column_order=False,
            use_gpu=False,
        )


@pytest.mark.order(374)
def test_get_score_raises_when_loaded_data_is_not_dataframe(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    data_path = tmp_path / "input.csv"
    data_path.write_text("x\n1\n", encoding="utf-8")
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: _PredictModel())
    monkeypatch.setattr(ocscoring.ocscoreio, "load_data", lambda *_a, **_k: ["not", "a", "dataframe"])

    with pytest.raises(ValueError, match="not a pandas DataFrame"):
        ocscoring.get_score(
            model_path=str(model_path),
            data=str(data_path),
            normalize=False,
            invert_conditionally=False,
            enforce_reference_column_order=False,
            use_gpu=False,
        )


@pytest.mark.order(375)
def test_get_score_raises_for_invalid_input_data_type(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: _PredictModel())

    with pytest.raises(ValueError, match="Data must be a pandas DataFrame"):
        ocscoring.get_score(
            model_path=str(model_path),
            data=123,  # type: ignore[arg-type]
            normalize=False,
            invert_conditionally=False,
            enforce_reference_column_order=False,
            use_gpu=False,
        )


@pytest.mark.order(376)
def test_get_score_raises_when_reference_column_order_is_missing(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: _PredictModel())
    fake_config_mod = types.ModuleType("OCDocker.Config")
    fake_config_mod.get_config = lambda: SimpleNamespace(paths=SimpleNamespace(reference_column_order=""))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.Config", fake_config_mod)

    with pytest.raises(ValueError, match="reference_column_order"):
        ocscoring.get_score(
            model_path=str(model_path),
            data=_base_df(),
            normalize=False,
            invert_conditionally=False,
            enforce_reference_column_order=True,
            use_gpu=False,
        )


@pytest.mark.order(377)
def test_get_score_raises_for_mask_length_mismatch(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: _PredictModel())

    with pytest.raises(ValueError, match="Mask length"):
        ocscoring.get_score(
            model_path=str(model_path),
            data=_base_df(),
            mask=[1],
            normalize=False,
            invert_conditionally=False,
            enforce_reference_column_order=False,
            use_gpu=False,
        )


@pytest.mark.order(378)
def test_get_score_predict_path_returns_dataframe(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: _PredictModel())

    out = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )

    assert isinstance(out, pd.DataFrame)
    assert "predicted_score" in out.columns
    assert out["predicted_score"].tolist() == [4.0, 6.0]
    assert out["receptor"].tolist() == ["rec1", "rec2"]


@pytest.mark.order(379)
def test_get_score_forward_path_handles_mask_and_tensor_prediction(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    model = _ForwardModel()
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model)

    out = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )

    assert isinstance(out, pd.DataFrame)
    assert out["predicted_score"].tolist() == [4.0, 6.0]
    assert not isinstance(model.mask, dict)
    assert isinstance(model.mask, (list, torch.Tensor))


@pytest.mark.order(380)
def test_get_score_raises_when_model_has_no_predict_or_forward(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: object())

    with pytest.raises(ValueError, match="predict or forward"):
        ocscoring.get_score(
            model_path=str(model_path),
            data=_base_df(),
            normalize=False,
            invert_conditionally=False,
            enforce_reference_column_order=False,
            use_gpu=False,
        )


@pytest.mark.order(381)
def test_get_score_raises_when_model_dict_does_not_contain_model_object(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: {"x": 1, "y": 2})

    with pytest.raises(ValueError, match="no model object found"):
        ocscoring.get_score(
            model_path=str(model_path),
            data=_base_df(),
            normalize=False,
            invert_conditionally=False,
            enforce_reference_column_order=False,
            use_gpu=False,
        )


@pytest.mark.order(382)
def test_get_score_scaler_path_validation_branches(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    scaler_path = _scaler_file(tmp_path)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda path, *_a, **_k: _PredictModel() if path == str(model_path) else {"bad": "scaler"})

    with pytest.raises(ValueError, match="valid scaler object"):
        ocscoring.get_score(
            model_path=str(model_path),
            data=_base_df(),
            scaler_path=str(scaler_path),
            normalize=True,
            invert_conditionally=False,
            enforce_reference_column_order=False,
            use_gpu=False,
        )


@pytest.mark.order(383)
def test_get_score_normalize_tuple_return_branch(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: _PredictModel())
    monkeypatch.setattr(ocscoring.ocscoredata, "norm_data", lambda df, **_k: (df, object()))

    out = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        normalize=True,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )

    assert isinstance(out, pd.DataFrame)
    assert out["predicted_score"].tolist() == [4.0, 6.0]


@pytest.mark.order(402)
def test_get_score_reads_db_data_and_adds_unknown_db_column(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    model = _CapturePredictModel()
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model)

    class _Complexes:
        allDescriptors = ["D1"]

    lig = SimpleNamespace(name="ligA")
    rec = SimpleNamespace(name="recA")
    db_row = SimpleNamespace(d1=2.5, ligand=lig, receptor=rec)

    class _Session:
        def query(self, _model):
            class _Query:
                @staticmethod
                def all():
                    return [db_row]
            return _Query()

    class _SessionCtx:
        def __enter__(self):
            return _Session()

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

    init_mod = types.ModuleType("OCDocker.Initialise")
    init_mod.session = lambda: _SessionCtx()  # type: ignore[attr-defined]
    db_mod = types.ModuleType("OCDocker.DB")
    db_mod.__path__ = []  # type: ignore[attr-defined]
    db_models_mod = types.ModuleType("OCDocker.DB.Models")
    db_models_mod.__path__ = []  # type: ignore[attr-defined]
    complexes_mod = types.ModuleType("OCDocker.DB.Models.Complexes")
    complexes_mod.Complexes = _Complexes  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "OCDocker.Initialise", init_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.DB", db_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.DB.Models", db_models_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.DB.Models.Complexes", complexes_mod)
    import OCDocker as ocpkg
    monkeypatch.setattr(ocpkg, "Initialise", init_mod, raising=False)
    monkeypatch.setattr(ocpkg, "DB", db_mod, raising=False)
    monkeypatch.setattr(db_mod, "Models", db_models_mod, raising=False)
    monkeypatch.setattr(db_models_mod, "Complexes", complexes_mod, raising=False)

    out = ocscoring.get_score(
        model_path=str(model_path),
        data=None,
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )

    assert isinstance(out, pd.DataFrame)
    assert out["receptor"].tolist() == ["recA"]
    assert out["ligand"].tolist() == ["ligA"]
    assert out["db"].tolist() == ["UNKNOWN"]
    assert out["predicted_score"].tolist() == [2.5]
    assert model.seen_shape == (1, 1)


@pytest.mark.order(403)
def test_get_score_db_session_missing_raises(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: _PredictModel())

    class _Complexes:
        allDescriptors = []

    init_mod = types.ModuleType("OCDocker.Initialise")
    init_mod.session = None  # type: ignore[attr-defined]
    db_mod = types.ModuleType("OCDocker.DB")
    db_mod.__path__ = []  # type: ignore[attr-defined]
    db_models_mod = types.ModuleType("OCDocker.DB.Models")
    db_models_mod.__path__ = []  # type: ignore[attr-defined]
    complexes_mod = types.ModuleType("OCDocker.DB.Models.Complexes")
    complexes_mod.Complexes = _Complexes  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "OCDocker.Initialise", init_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.DB", db_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.DB.Models", db_models_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.DB.Models.Complexes", complexes_mod)
    import OCDocker as ocpkg
    monkeypatch.setattr(ocpkg, "Initialise", init_mod, raising=False)
    monkeypatch.setattr(ocpkg, "DB", db_mod, raising=False)
    monkeypatch.setattr(db_mod, "Models", db_models_mod, raising=False)
    monkeypatch.setattr(db_models_mod, "Complexes", complexes_mod, raising=False)

    with pytest.raises(ValueError, match="Database session not available"):
        ocscoring.get_score(
            model_path=str(model_path),
            data=None,
            normalize=False,
            invert_conditionally=False,
            enforce_reference_column_order=False,
            use_gpu=False,
        )


@pytest.mark.order(404)
def test_get_score_db_empty_result_raises(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: _PredictModel())

    class _Complexes:
        allDescriptors = ["D1"]

    class _Session:
        def query(self, _model):
            class _Query:
                @staticmethod
                def all():
                    return []
            return _Query()

    class _SessionCtx:
        def __enter__(self):
            return _Session()

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

    init_mod = types.ModuleType("OCDocker.Initialise")
    init_mod.session = lambda: _SessionCtx()  # type: ignore[attr-defined]
    db_mod = types.ModuleType("OCDocker.DB")
    db_mod.__path__ = []  # type: ignore[attr-defined]
    db_models_mod = types.ModuleType("OCDocker.DB.Models")
    db_models_mod.__path__ = []  # type: ignore[attr-defined]
    complexes_mod = types.ModuleType("OCDocker.DB.Models.Complexes")
    complexes_mod.Complexes = _Complexes  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "OCDocker.Initialise", init_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.DB", db_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.DB.Models", db_models_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.DB.Models.Complexes", complexes_mod)
    import OCDocker as ocpkg
    monkeypatch.setattr(ocpkg, "Initialise", init_mod, raising=False)
    monkeypatch.setattr(ocpkg, "DB", db_mod, raising=False)
    monkeypatch.setattr(db_mod, "Models", db_models_mod, raising=False)
    monkeypatch.setattr(db_models_mod, "Complexes", complexes_mod, raising=False)

    with pytest.raises(ValueError, match="No data found in database"):
        ocscoring.get_score(
            model_path=str(model_path),
            data=None,
            normalize=False,
            invert_conditionally=False,
            enforce_reference_column_order=False,
            use_gpu=False,
        )


@pytest.mark.order(405)
def test_get_score_raises_when_scaler_file_does_not_exist(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: _PredictModel())

    with pytest.raises(FileNotFoundError, match="Scaler file not found"):
        ocscoring.get_score(
            model_path=str(model_path),
            data=_base_df(),
            scaler_path=str(tmp_path / "missing_scaler.bin"),
            normalize=True,
            invert_conditionally=False,
            enforce_reference_column_order=False,
            use_gpu=False,
        )


@pytest.mark.order(406)
def test_get_score_wraps_scaler_loading_errors(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    scaler_path = _scaler_file(tmp_path)

    def _load_object(path, *_a, **_k):
        if path == str(model_path):
            return _PredictModel()
        raise RuntimeError("cannot load scaler")

    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", _load_object)

    with pytest.raises(ValueError, match="Failed to load scaler"):
        ocscoring.get_score(
            model_path=str(model_path),
            data=_base_df(),
            scaler_path=str(scaler_path),
            normalize=True,
            invert_conditionally=False,
            enforce_reference_column_order=False,
            use_gpu=False,
        )


@pytest.mark.order(407)
def test_get_score_applies_pca_and_valid_external_mask(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    model = _CapturePredictModel()
    calls = {"pca": 0}

    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model)
    monkeypatch.setattr(
        ocscoring.ocscoredata,
        "apply_pca",
        lambda df, *_a, **_k: calls.__setitem__("pca", calls["pca"] + 1) or df.assign(pc3=df["feat1"] + df["feat2"]),
    )

    out = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        pca_model=object(),
        mask=[1, 0, 1],
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )

    assert isinstance(out, pd.DataFrame)
    assert calls["pca"] == 1
    assert model.seen_shape == (2, 2)


@pytest.mark.order(408)
def test_get_score_handles_no_scores_and_only_scores_paths(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    model_no_scores = _CapturePredictModel()
    model_only_scores = _CapturePredictModel()

    df = pd.DataFrame(
        {
            "receptor": ["rec1", "rec2"],
            "ligand": ["lig1", "lig2"],
            "VINA_A": [1.0, 2.0],
            "SMINA_A": [1.5, 2.5],
            "ODDT_A": [0.5, 1.5],
            "PLANTS_A": [0.2, 0.3],
            "feat1": [3.0, 4.0],
            "feat2": [5.0, 6.0],
        }
    )

    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model_no_scores)
    _ = ocscoring.get_score(
        model_path=str(model_path),
        data=df,
        no_scores=True,
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )

    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model_only_scores)
    _ = ocscoring.get_score(
        model_path=str(model_path),
        data=df,
        only_scores=True,
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )

    assert model_no_scores.seen_shape == (2, 2)
    assert model_only_scores.seen_shape == (2, 4)


@pytest.mark.order(409)
def test_get_score_forward_path_converts_unexpected_mask_dict_to_empty_list(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    model = _ForwardFallbackMaskModel()
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model)

    out = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )

    assert isinstance(out, pd.DataFrame)
    assert out["predicted_score"].tolist() == [4.0, 6.0]
    assert model.mask == []


@pytest.mark.order(410)
def test_get_score_enforce_reference_column_order_propagates_reorder_errors(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: _PredictModel())
    fake_config_mod = types.ModuleType("OCDocker.Config")
    fake_config_mod.get_config = lambda: SimpleNamespace(paths=SimpleNamespace(reference_column_order="set"))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.Config", fake_config_mod)
    monkeypatch.setattr(
        ocscoring.ocscoredata,
        "reorder_columns_to_match_data_order",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("reorder failed")),
    )

    with pytest.raises(RuntimeError, match="reorder failed"):
        ocscoring.get_score(
            model_path=str(model_path),
            data=_base_df(),
            normalize=False,
            invert_conditionally=False,
            enforce_reference_column_order=True,
            use_gpu=False,
        )
