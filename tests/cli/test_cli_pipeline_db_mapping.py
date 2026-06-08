#!/usr/bin/env python3

# Description
###############################################################################
'''
Unit tests for CLI pipeline DB score mapping helpers.
'''

# Imports
###############################################################################
from __future__ import annotations

from OCDocker.CLI.workflow import _flatten_rescoring_to_complex_payload, _map_score_to_complex_column


def test_map_score_to_complex_column_supported_keys():
    assert _map_score_to_complex_column("vina_vina") == "VINA_VINA"
    assert _map_score_to_complex_column("smina_dkoes_scoring") == "SMINA_SCORING_DKOES"
    assert _map_score_to_complex_column("smina_dkoes_scoring_old") == "SMINA_OLD_SCORING_DKOES"
    assert _map_score_to_complex_column("gnina_default") == "GNINA_DEFAULT"
    assert _map_score_to_complex_column("gnina_scoring_ad4") == "GNINA_AD4_SCORING"
    assert _map_score_to_complex_column("gnina_old_scoring_dkoes") == "GNINA_DKOES_SCORING_OLD"
    assert _map_score_to_complex_column("plants_plp95") == "PLANTS_PLP95"
    assert _map_score_to_complex_column("oddt_rfscore_v2_pdbbind2016") == "ODDT_RFSCORE_V2"
    assert _map_score_to_complex_column("oddt_plecrf_p5_l1_s65536") == "ODDT_PLECRF_P5_L1_S65536"
    assert _map_score_to_complex_column("oddt_nnscore_v1") == "ODDT_NNSCORE"


def test_map_score_to_complex_column_unsupported_key():
    assert _map_score_to_complex_column("gnina_cnn_default") is None


def test_flatten_rescoring_to_complex_payload_maps_and_ignores():
    rescoring = {
        "vina": {"vina_vina": -7.1},
        "smina": {"smina_dkoes_scoring": -8.2, "invalid_non_numeric": "x"},
        "gnina": {"gnina_default": -9.0},
        "oddt": {"oddt_rfscore_v1_pdbbind2016": 1.23},
        "gnina_cnn": {"gnina_cnn_default": -7.7},
    }

    payload, ignored = _flatten_rescoring_to_complex_payload(rescoring)

    assert payload["VINA_VINA"] == -7.1
    assert payload["SMINA_SCORING_DKOES"] == -8.2
    assert payload["GNINA_DEFAULT"] == -9.0
    assert payload["ODDT_RFSCORE_V1"] == 1.23
    assert "gnina_cnn_default" in ignored
