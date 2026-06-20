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
from sklearn.preprocessing import StandardScaler

# License
###############################################################################
'''OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

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


class _PredictModelWithToNoDevice:
    def eval(self):
        return self

    def to(self, _device):
        return self

    def predict(self, x):
        x_arr = np.asarray(x, dtype=float)
        return x_arr.sum(axis=1)


class _MaskCarrierPredictModel:
    def __init__(self, mask):
        self.mask = mask

    def predict(self, x):
        x_arr = np.asarray(x, dtype=float)
        return x_arr.sum(axis=1)


class _RecursiveMaskCarrierModel(_MaskCarrierPredictModel):
    def __init__(self, mask, child):
        super().__init__(mask)
        self.child = child

    def modules(self):
        return [self, self.child]


class _ForwardMaskModel:
    def __init__(self, mask, *, freeze_mask=False, return_numpy=False):
        self._mask = mask
        self._freeze_mask = freeze_mask
        self._return_numpy = return_numpy

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        if not self._freeze_mask:
            self._mask = value

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def forward(self, x):
        if self._return_numpy:
            return np.arange(x.shape[0], dtype=float)
        return x.sum(dim=1, keepdim=True)

    __call__ = forward


class _ForwardNoMaskModel:
    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def forward(self, x):
        return x.sum(dim=1, keepdim=True)

    __call__ = forward


class _LenOneMask:
    def __len__(self):
        return 1


class _TwoPhaseValuesDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._values_calls = 0

    def values(self):
        self._values_calls += 1
        if self._values_calls < 3:
            return [123]
        return [np.array([1.0, 0.0])]


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


@pytest.mark.order(411)
@pytest.mark.parametrize(
    "payload",
    [
        {"model": _PredictModel()},
        {"network": _PredictModel()},
        {"wrapped": _PredictModel()},
    ],
)
def test_get_score_loads_model_from_dict_variants(monkeypatch, tmp_path, payload):
    model_path = _model_file(tmp_path)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: payload)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    out = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=True,
    )

    assert isinstance(out, pd.DataFrame)
    assert "predicted_score" in out.columns


@pytest.mark.order(412)
def test_get_score_to_without_device_attribute_branch(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    model = _PredictModelWithToNoDevice()
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


@pytest.mark.order(413)
def test_get_score_fix_mask_attribute_variants(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)

    # mask is None -> []
    model_none = _MaskCarrierPredictModel(mask=None)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model_none)
    _ = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )
    assert model_none.mask == []

    # dict with tensor-like value found in generic values loop
    model_tensor = _MaskCarrierPredictModel(mask={"unexpected": torch.tensor([1.0, 0.0])})
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model_tensor)
    _ = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )
    assert isinstance(model_tensor.mask, torch.Tensor)

    # dict with ndarray from generic values loop
    model_ndarray = _MaskCarrierPredictModel(mask={"unexpected": np.array([1.0, 0.0])})
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model_ndarray)
    _ = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )
    assert isinstance(model_ndarray.mask, torch.Tensor)

    # dict with common key but unsupported value -> []
    model_unknown = _MaskCarrierPredictModel(mask={"mask": object()})
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model_unknown)
    _ = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )
    assert model_unknown.mask == []

    # ndarray/list direct mask conversion branch
    model_direct = _MaskCarrierPredictModel(mask=np.array([1.0, 0.0]))
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model_direct)
    _ = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )
    assert isinstance(model_direct.mask, torch.Tensor)


@pytest.mark.order(414)
def test_get_score_fix_mask_attribute_recurses_nested_modules(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    child = _MaskCarrierPredictModel(mask=[1.0, 0.0])
    model = _RecursiveMaskCarrierModel(mask=[], child=child)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model)

    _ = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )

    assert isinstance(child.mask, torch.Tensor)


@pytest.mark.order(415)
def test_get_score_db_branches_for_missing_fields_and_existing_db(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    model = _CapturePredictModel()
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model)

    class _Complexes:
        allDescriptors = ["D1", "D2", "db"]

    row1 = SimpleNamespace(
        d1=None,  # hits value-is-None branch
        db="KNOWN_DB",  # keeps db present and skips UNKNOWN assignment
        ligand=SimpleNamespace(),  # no name
        receptor=None,  # receptor condition false
    )
    row2 = SimpleNamespace(
        d1=1.0,
        db="KNOWN_DB",
        receptor=SimpleNamespace(),  # receptor exists but no name
    )

    class _Session:
        def query(self, _model):
            class _Query:
                @staticmethod
                def all():
                    return [row1, row2]
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
    assert out["db"].tolist() == ["KNOWN_DB", "KNOWN_DB"]


@pytest.mark.order(416)
def test_get_score_db_import_or_attribute_error_is_wrapped(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: _PredictModel())

    class _Complexes:
        allDescriptors = []

    class _BrokenSessionCtx:
        def __enter__(self):
            raise AttributeError("broken session")

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

    init_mod = types.ModuleType("OCDocker.Initialise")
    init_mod.session = lambda: _BrokenSessionCtx()  # type: ignore[attr-defined]
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

    with pytest.raises(ValueError, match="Failed to read from database"):
        ocscoring.get_score(
            model_path=str(model_path),
            data=None,
            normalize=False,
            invert_conditionally=False,
            enforce_reference_column_order=False,
            use_gpu=False,
        )


@pytest.mark.order(417)
def test_get_score_data_path_success_assigns_loaded_dataframe(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    data_path = tmp_path / "loaded.csv"
    data_path.write_text("x\n1\n", encoding="utf-8")
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: _PredictModel())
    monkeypatch.setattr(ocscoring.ocscoreio, "load_data", lambda *_a, **_k: _base_df())

    out = ocscoring.get_score(
        model_path=str(model_path),
        data=str(data_path),
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )

    assert isinstance(out, pd.DataFrame)
    assert out["predicted_score"].tolist() == [4.0, 6.0]


@pytest.mark.order(418)
def test_get_score_score_columns_empty_and_invert_branch(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    model = _CapturePredictModel()
    invert_calls = {"n": 0}

    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model)
    monkeypatch.setattr(
        ocscoring.ocscoredata,
        "invert_values_conditionally",
        lambda df, **_k: invert_calls.__setitem__("n", invert_calls["n"] + 1) or df,
    )

    out = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        score_columns_list=[],
        no_scores=True,
        normalize=False,
        invert_conditionally=True,
        enforce_reference_column_order=False,
        use_gpu=False,
    )

    assert isinstance(out, pd.DataFrame)
    assert invert_calls["n"] == 1
    assert model.seen_shape == (2, 2)


@pytest.mark.order(419)
def test_get_score_uses_valid_loaded_scaler_object(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    scaler_path = _scaler_file(tmp_path)
    scaler_obj = StandardScaler()
    captured = {"scaler": None}

    def _load_object(path, *_a, **_k):
        if path == str(model_path):
            return _PredictModel()
        return scaler_obj

    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", _load_object)
    monkeypatch.setattr(
        ocscoring.ocscoredata,
        "norm_data",
        lambda df, scaler=None, **_k: captured.__setitem__("scaler", scaler) or df,
    )

    out = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        scaler_path=str(scaler_path),
        normalize=True,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )

    assert isinstance(out, pd.DataFrame)
    assert captured["scaler"] is scaler_obj


@pytest.mark.order(420)
def test_get_score_normalize_non_tuple_return_path(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: _PredictModel())
    monkeypatch.setattr(ocscoring.ocscoredata, "norm_data", lambda df, **_k: df)

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


@pytest.mark.order(421)
def test_get_score_pca_default_skip_extends_with_score_columns(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    model = _CapturePredictModel()
    seen = {"skip": None}
    df = pd.DataFrame(
        {
            "receptor": ["r1", "r2"],
            "ligand": ["l1", "l2"],
            "VINA_SCORE": [1.0, 2.0],
            "feat1": [3.0, 4.0],
        }
    )
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model)
    monkeypatch.setattr(
        ocscoring.ocscoredata,
        "apply_pca",
        lambda frame, _pca, columns_to_skip_pca=None, **_k: seen.__setitem__("skip", list(columns_to_skip_pca or [])) or frame,
    )

    out = ocscoring.get_score(
        model_path=str(model_path),
        data=df,
        pca_model=object(),
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )

    assert isinstance(out, pd.DataFrame)
    assert "VINA_SCORE" in (seen["skip"] or [])


@pytest.mark.order(422)
def test_get_score_pca_with_explicit_columns_to_skip(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    seen = {"skip": None}
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: _PredictModel())
    monkeypatch.setattr(
        ocscoring.ocscoredata,
        "apply_pca",
        lambda frame, _pca, columns_to_skip_pca=None, **_k: seen.__setitem__("skip", list(columns_to_skip_pca or [])) or frame,
    )

    _ = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        pca_model=object(),
        columns_to_skip_pca=["receptor"],
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )

    assert seen["skip"] == ["receptor"]


@pytest.mark.order(423)
def test_get_score_forward_paths_without_mask_and_with_mask_none(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)

    model_no_mask = _ForwardNoMaskModel()
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model_no_mask)
    out_no_mask = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )
    assert isinstance(out_no_mask, pd.DataFrame)

    model_mask_none = _ForwardMaskModel(mask=None)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model_mask_none)
    out_mask_none = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )
    assert isinstance(out_mask_none, pd.DataFrame)
    assert model_mask_none.mask == []


@pytest.mark.order(424)
@pytest.mark.parametrize(
    "frozen_mask",
    [
        {"other": np.array([1.0, 0.0])},  # values loop + ndarray conversion path
        {"other": torch.tensor([1.0, 0.0])},  # values loop + tensor conversion path
        {"other": 5},  # first-value fallback to []
        {"mask": object()},  # unsupported extracted value -> []
    ],
)
def test_get_score_forward_dict_mask_conversion_paths(monkeypatch, tmp_path, frozen_mask):
    model_path = _model_file(tmp_path)
    model = _ForwardMaskModel(mask=frozen_mask, freeze_mask=True)
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
    # Frozen setter keeps dict alive, so final dict safety branch is exercised.
    assert isinstance(model.mask, dict)


@pytest.mark.order(425)
def test_get_score_forward_mask_scalar_and_numpy_prediction_paths(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    model = _ForwardMaskModel(mask=_LenOneMask(), return_numpy=True)
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
    assert out["predicted_score"].tolist() == [0.0, 1.0]


@pytest.mark.order(426)
def test_get_score_returns_dataframe_without_metadata_columns(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    df = pd.DataFrame({"feat1": [1.0, 2.0], "feat2": [3.0, 4.0]})
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: _PredictModel())

    out = ocscoring.get_score(
        model_path=str(model_path),
        data=df,
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["predicted_score"]


@pytest.mark.order(427)
def test_get_score_fix_mask_attribute_list_empty_branch(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    model = _MaskCarrierPredictModel(mask=[])
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
    assert model.mask == []


@pytest.mark.order(428)
def test_get_score_forward_freeze_mask_none_and_list_conversion(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)

    model_none = _ForwardMaskModel(mask=None, freeze_mask=True)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model_none)
    out_none = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )
    assert isinstance(out_none, pd.DataFrame)

    model_list = _ForwardMaskModel(mask={"mask": [1.0, 0.0]}, freeze_mask=True)
    monkeypatch.setattr(ocscoring.ocscoreio, "load_object", lambda *_a, **_k: model_list)
    out_list = ocscoring.get_score(
        model_path=str(model_path),
        data=_base_df(),
        normalize=False,
        invert_conditionally=False,
        enforce_reference_column_order=False,
        use_gpu=False,
    )
    assert isinstance(out_list, pd.DataFrame)


@pytest.mark.order(429)
def test_get_score_forward_dict_first_value_array_like_branch(monkeypatch, tmp_path):
    model_path = _model_file(tmp_path)
    model = _ForwardMaskModel(mask=_TwoPhaseValuesDict({"a": "b"}), freeze_mask=True)
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
