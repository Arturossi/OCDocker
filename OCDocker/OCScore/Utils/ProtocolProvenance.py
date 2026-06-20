#!/usr/bin/env python3

# Description
###############################################################################
'''Production-grade provenance artifact bundle for staged OCScore runs.'''

# Imports
###############################################################################
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import OCDocker.Toolbox.Reproducibility as ocrepro


def build_split_assignments_payload(
        *,
        pdbbind_stage: Optional[dict[str, Any]] = None,
        dudez_stage: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
    '''Build split assignment payload for ``split_assignments.json``.

    Parameters
    ----------
    pdbbind_stage : dict[str, Any] | None, optional
        PDBbind stage summary from ``context.stage_results``.
    dudez_stage : dict[str, Any] | None, optional
        DUDEz stage summary from ``context.stage_results``.

    Returns
    -------
    dict[str, Any]
        JSON-serializable split indices and diagnostics.
    '''

    payload: dict[str, Any] = {}
    if pdbbind_stage:
        pdb_payload: dict[str, Any] = {}
        if "split_indices" in pdbbind_stage:
            pdb_payload["split_indices"] = pdbbind_stage["split_indices"]
        diagnostics = pdbbind_stage.get("split_diagnostics") or {}
        if diagnostics:
            pdb_payload["split_diagnostics"] = {
                "strategy": diagnostics.get("strategy"),
                "random_seed": diagnostics.get("random_seed"),
                "receptor_counts": diagnostics.get("receptor_counts"),
                "receptor_overlap": diagnostics.get("receptor_overlap"),
            }
        if pdb_payload:
            payload["pdbbind"] = pdb_payload
    if dudez_stage:
        dudez_payload: dict[str, Any] = {}
        if "split_indices" in dudez_stage:
            dudez_payload["split_indices"] = dudez_stage["split_indices"]
        diagnostics = dudez_stage.get("split_diagnostics") or {}
        if diagnostics:
            dudez_payload["split_diagnostics"] = {
                "strategy": diagnostics.get("strategy"),
                "random_seed": diagnostics.get("random_seed"),
                "train_receptors": diagnostics.get("train_receptors"),
                "validation_receptors": diagnostics.get("validation_receptors"),
                "test_receptors": diagnostics.get("test_receptors"),
                "splits": diagnostics.get("splits"),
            }
        if dudez_payload:
            payload["dudez"] = dudez_payload
    return payload

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def write_production_provenance_bundle(
        output_dir: str | Path,
        *,
        feature_selection: Optional[dict[str, Any]] = None,
        scaling: Optional[dict[str, Any]] = None,
        split_assignments: Optional[dict[str, Any]] = None,
        data_provenance: Optional[dict[str, Any]] = None,
        command: Optional[dict[str, Any]] = None,
        final_report: Optional[dict[str, Any]] = None,
        leakage_audit: Optional[dict[str, Any]] = None,
    ) -> dict[str, str]:
    '''Write standard production-grade provenance JSON artifacts.

    Returns
    -------
    dict[str, str]
        Mapping from artifact name to written path.
    '''

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    timestamp = datetime.now(timezone.utc).isoformat()

    if feature_selection is not None:
        paths["feature_selection"] = _write_json(out / "feature_selection.json", feature_selection)
    if scaling is not None:
        paths["scaling"] = _write_json(out / "scaling.json", scaling)
    if split_assignments is not None:
        paths["split_assignments"] = _write_json(out / "split_assignments.json", split_assignments)
    if data_provenance is not None:
        paths["data_provenance"] = _write_json(out / "data_provenance.json", data_provenance)

    environment = {
        "timestamp_utc": timestamp,
        "python_version": sys.version,
        "platform": platform.platform(),
        "reproducibility": ocrepro.generate_reproducibility_manifest(include_python_packages=True),
    }
    paths["environment"] = _write_json(out / "environment.json", environment)

    command_payload = {"timestamp_utc": timestamp, **(command or {})}
    paths["command"] = _write_json(out / "command.json", command_payload)

    if final_report is not None:
        paths["final_report"] = _write_json(out / "final_report.json", final_report)
    if leakage_audit is not None:
        paths["leakage_audit"] = _write_json(out / "leakage_audit.json", leakage_audit)

    return paths


__all__ = ["build_split_assignments_payload", "write_production_provenance_bundle"]
