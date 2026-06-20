#!/usr/bin/env python3

# Description
###############################################################################
'''
Serialization helpers for OCDocker workbench specs and manifests.
'''

# Imports
###############################################################################
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from OCDocker.Workbench.Models import OCScoreAblationSpec
from OCDocker.Workbench.Models import OCScoreStudySpec
from OCDocker.Workbench.Models import ResultManifest
from OCDocker.Workbench.Models import RunManifest
from OCDocker.Workbench.Models import VSCampaignSpec
from OCDocker.Workbench.Models import WorkbenchModel
from OCDocker.Workbench.Models import WorkbenchSpec

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

# Constants
###############################################################################

_SPEC_MODELS = {
    "vs_campaign": VSCampaignSpec,
    "ocscore_study": OCScoreStudySpec,
    "ocscore_ablation": OCScoreAblationSpec,
}


# Functions
###############################################################################
## Public ##


def model_to_data(model: WorkbenchModel) -> dict[str, Any]:
    '''Return a JSON-compatible dictionary for a workbench model.

    Parameters
    ----------
    model : WorkbenchModel
        Input value.

    Returns
    -------
    dict[str, Any]
        Returned value.
    '''

    return model.model_dump(mode="json", exclude_none=True)


def write_model(path: str | Path, model: WorkbenchModel) -> Path:
    '''Write a workbench model to JSON or YAML based on the file suffix.

    Parameters
    ----------
    path : str | Path
        Input value.
    model : WorkbenchModel
        Input value.

    Returns
    -------
    Path
        Returned value.
    '''

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = model_to_data(model)
    if output_path.suffix.lower() == ".json":
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        output_path.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
    return output_path


def read_data(path: str | Path) -> dict[str, Any]:
    '''Read a JSON or YAML mapping from disk.

    Parameters
    ----------
    path : str | Path
        Input value.

    Returns
    -------
    dict[str, Any]
        Returned value.
    '''

    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if input_path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in workbench file: {input_path}")
    return loaded


def read_spec(path: str | Path) -> WorkbenchSpec:
    '''Load and validate a workbench spec from JSON or YAML.

    Parameters
    ----------
    path : str | Path
        Input value.

    Returns
    -------
    WorkbenchSpec
        Returned value.
    '''

    payload = read_data(path)
    spec_type = str(payload.get("type", "")).strip()
    model = _SPEC_MODELS.get(spec_type)
    if model is None:
        valid = ", ".join(sorted(_SPEC_MODELS))
        raise ValueError(
            f"Unknown workbench spec type {spec_type!r}. Expected one of: {valid}."
        )
    return model.model_validate(payload)


def read_run_manifest(path: str | Path) -> RunManifest:
    '''Load and validate a run manifest from JSON or YAML.

    Parameters
    ----------
    path : str | Path
        Input value.

    Returns
    -------
    RunManifest
        Returned value.
    '''

    return RunManifest.model_validate(read_data(path))


def read_result_manifest(path: str | Path) -> ResultManifest:
    '''Load and validate a result manifest from JSON or YAML.

    Parameters
    ----------
    path : str | Path
        Input value.

    Returns
    -------
    ResultManifest
        Returned value.
    '''

    return ResultManifest.model_validate(read_data(path))


__all__ = [
    "model_to_data",
    "read_data",
    "read_result_manifest",
    "read_run_manifest",
    "read_spec",
    "write_model",
]
