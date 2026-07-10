#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Analysis.AblationSignificance.
'''

# Imports
###############################################################################
import numpy as np
import pandas as pd

import pytest

from OCDocker.OCScore.Analysis.AblationSignificance import (
    AblationSignificanceConfig,
    build_ablation_bedroc_significance_table,
    compute_ablation_significance,
    load_ablation_bedroc_long,
)

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

def _write_replicas_csv(tmp_path, name: str, seeds, values) -> str:
    path = tmp_path / f"{name}_replicas_summary.csv"
    pd.DataFrame({
        "replica_index": list(range(len(seeds))),
        "replica_name": [f"replica_{i:03d}" for i in range(len(seeds))],
        "seed": seeds,
        "dudez_test_bedroc": values,
    }).to_csv(path, index=False)
    return str(path)


def _write_ablation_summary_csv(tmp_path, policies_to_values: dict[str, list[float]], *, seeds=None) -> str:
    seeds = seeds or [42, 43, 44, 45, 46]
    rows = []
    for policy_name, values in policies_to_values.items():
        replicas_csv = _write_replicas_csv(tmp_path, policy_name, seeds, values)
        rows.append({"feature_policy_name": policy_name, "replicas_summary_csv": replicas_csv})
    summary_path = tmp_path / "ablation_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return str(summary_path)


## Public ##

@pytest.mark.order(504)
def test_load_ablation_bedroc_long_reads_matched_seeds(tmp_path):
    summary_csv = _write_ablation_summary_csv(
        tmp_path,
        {
            "full_ocscore": [0.40, 0.43, 0.42, 0.41, 0.44],
            "shape_only": [0.16, 0.17, 0.15, 0.18, 0.16],
        },
    )

    long_df = load_ablation_bedroc_long(summary_csv)

    assert set(long_df["policy"].unique()) == {"full_ocscore", "shape_only"}
    assert list(long_df.columns) == ["policy", "replica_index", "seed", "dudez_test_bedroc"]
    assert len(long_df) == 10


@pytest.mark.order(505)
def test_load_ablation_bedroc_long_raises_on_missing_replicas_csv(tmp_path):
    summary_path = tmp_path / "ablation_summary.csv"
    pd.DataFrame([{"feature_policy_name": "full_ocscore", "replicas_summary_csv": str(tmp_path / "missing.csv")}]).to_csv(
        summary_path, index=False
    )

    with pytest.raises(FileNotFoundError):
        load_ablation_bedroc_long(summary_path)


@pytest.mark.order(506)
def test_load_ablation_bedroc_long_raises_on_missing_columns(tmp_path):
    replicas_path = tmp_path / "bad_replicas.csv"
    pd.DataFrame({"replica_index": [0], "seed": [42]}).to_csv(replicas_path, index=False)
    summary_path = tmp_path / "ablation_summary.csv"
    pd.DataFrame([{"feature_policy_name": "full_ocscore", "replicas_summary_csv": str(replicas_path)}]).to_csv(
        summary_path, index=False
    )

    with pytest.raises(ValueError, match="missing required column"):
        load_ablation_bedroc_long(summary_path)


@pytest.mark.order(507)
def test_compute_ablation_significance_flags_large_and_small_effects():
    replica_long_df = pd.DataFrame({
        "policy": ["full_ocscore"] * 5 + ["shape_only"] * 5 + ["no_pmi"] * 5,
        "replica_index": list(range(5)) * 3,
        "seed": [42, 43, 44, 45, 46] * 3,
        "dudez_test_bedroc": (
            [0.40, 0.43, 0.42, 0.41, 0.44]
            + [0.16, 0.17, 0.15, 0.18, 0.16]
            + [0.41, 0.41, 0.435, 0.405, 0.46]
        ),
    })

    result = compute_ablation_significance(replica_long_df)

    assert set(result["policy"]) == {"shape_only", "no_pmi"}
    shape_only_row = result.loc[result["policy"] == "shape_only"].iloc[0]
    no_pmi_row = result.loc[result["policy"] == "no_pmi"].iloc[0]

    assert shape_only_row["reject_null"]
    assert shape_only_row["direction"] == "worse"
    assert shape_only_row["pvalue_corrected"] < 0.05
    assert not no_pmi_row["reject_null"]
    assert no_pmi_row["direction"] == "not_significant"
    # Holm correction must never make a corrected p-value smaller than its raw p-value.
    assert (result["pvalue_corrected"] >= result["pvalue"]).all()


@pytest.mark.order(508)
def test_compute_ablation_significance_raises_when_reference_missing():
    replica_long_df = pd.DataFrame({
        "policy": ["shape_only"] * 5,
        "replica_index": list(range(5)),
        "seed": [42, 43, 44, 45, 46],
        "dudez_test_bedroc": [0.16, 0.17, 0.15, 0.18, 0.16],
    })

    with pytest.raises(ValueError, match="Reference policy"):
        compute_ablation_significance(replica_long_df)


@pytest.mark.order(509)
def test_compute_ablation_significance_wilcoxon_method_runs():
    replica_long_df = pd.DataFrame({
        "policy": ["full_ocscore"] * 5 + ["shape_only"] * 5,
        "replica_index": list(range(5)) * 2,
        "seed": [42, 43, 44, 45, 46] * 2,
        "dudez_test_bedroc": (
            [0.40, 0.43, 0.42, 0.41, 0.44]
            + [0.16, 0.17, 0.15, 0.18, 0.16]
        ),
    })

    result = compute_ablation_significance(
        replica_long_df,
        config=AblationSignificanceConfig(method="wilcoxon"),
    )

    assert len(result) == 1
    assert np.isfinite(result.loc[0, "pvalue"])
    # n=5 two-sided Wilcoxon can never reach p<0.05 (exact minimum is 0.0625).
    assert not result.loc[0, "reject_null"]


@pytest.mark.order(510)
def test_compute_ablation_significance_rejects_unknown_method():
    replica_long_df = pd.DataFrame({
        "policy": ["full_ocscore", "shape_only"],
        "replica_index": [0, 0],
        "seed": [42, 42],
        "dudez_test_bedroc": [0.40, 0.16],
    })

    with pytest.raises(ValueError, match="Unknown paired test method"):
        compute_ablation_significance(
            replica_long_df,
            config=AblationSignificanceConfig(method="not_a_real_method"),
        )


@pytest.mark.order(511)
def test_build_ablation_bedroc_significance_table_end_to_end(tmp_path):
    summary_csv = _write_ablation_summary_csv(
        tmp_path,
        {
            "full_ocscore": [0.40, 0.43, 0.42, 0.41, 0.44],
            "shape_only": [0.16, 0.17, 0.15, 0.18, 0.16],
            "no_pmi": [0.41, 0.41, 0.435, 0.405, 0.46],
        },
    )

    result = build_ablation_bedroc_significance_table(summary_csv)

    assert len(result) == 2
    assert result.iloc[0]["policy"] == "shape_only"
    assert result.iloc[0]["pvalue_corrected"] < result.iloc[1]["pvalue_corrected"]
