#!/usr/bin/env python3

# Description
###############################################################################
'''Tests for feature-selection scope metadata and train-only guards.'''

# Imports
###############################################################################
import json

import pytest

from OCDocker.OCScore.Utils.FeatureSelectionMetadata import FeatureSelectionScope
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import attach_feature_hashes
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import load_feature_selection_json
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import validate_train_only_feature_selection
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import write_feature_selection_fit_rows_json
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import write_feature_selection_json

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''


@pytest.mark.order(401)
def test_precomputed_global_scope_round_trip(tmp_path):
    scope = FeatureSelectionScope.precomputed_global(n_selected_features=12)
    path = write_feature_selection_json(tmp_path, scope)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["scope"] == "precomputed_global"
    assert payload["fit_dataset"] == "merged_pdbbind_dudez"
    assert payload["uses_supervised_target"] is False


@pytest.mark.order(402)
def test_train_only_validation_rejects_precomputed_global():
    scope = FeatureSelectionScope.precomputed_global()
    with pytest.raises(ValueError, match="Staged train requires train-only"):
        validate_train_only_feature_selection(scope)


@pytest.mark.order(403)
def test_missing_feature_selection_json_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Feature selection metadata not found"):
        load_feature_selection_json(tmp_path)


@pytest.mark.order(404)
def test_precomputed_scope_includes_feature_hashes(tmp_path):
    scope = FeatureSelectionScope.precomputed_global(n_selected_features=2)
    scope = attach_feature_hashes(scope, selected_features=["a", "b"], removed_features=["c"])
    path = write_feature_selection_json(tmp_path, scope)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["feature_selection_scope"] == "precomputed_global"
    assert payload["selected_features_hash"]
    assert payload["removed_features_hash"]


@pytest.mark.order(405)
def test_validate_train_only_feature_selection_accepts_train_scope():
    scope = FeatureSelectionScope.train_only(
        fit_row_count=10,
        selected_features=["f1", "f2"],
    )
    validate_train_only_feature_selection(scope)

@pytest.mark.order(406)
def test_train_only_scope_omits_fit_row_indices_from_summary_by_default(tmp_path):
    scope = FeatureSelectionScope.train_only(
        fit_row_count=4,
        fit_row_indices=[10, 11, 12, 13],
        selected_features=["f1"],
    )

    summary = scope.to_dict()
    assert "fit_row_indices" not in summary
    assert summary["fit_row_count"] == 4
    assert summary["fit_row_indices_hash"]

    full = scope.to_dict(include_fit_row_indices=True)
    assert full["fit_row_indices"] == [10, 11, 12, 13]

    fit_rows_path = write_feature_selection_fit_rows_json(tmp_path, scope)
    payload = json.loads(fit_rows_path.read_text(encoding="utf-8"))
    assert payload["fit_row_indices"] == [10, 11, 12, 13]
    assert payload["fit_row_indices_hash"] == summary["fit_row_indices_hash"]
    assert scope.fit_row_indices_artifact == str(fit_rows_path)
