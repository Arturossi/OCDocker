#!/usr/bin/env python3

# Description
###############################################################################
'''
Interactive ablation-design helpers for the strict OCScore Workbench dashboard.
'''

# Imports
###############################################################################
from __future__ import annotations

import json

from pathlib import Path
from typing import Any
from typing import Optional

from OCDocker.Workbench.IO import model_to_data
from OCDocker.Workbench.Models import FeaturePolicySelection
from OCDocker.Workbench.Models import OCScoreAblationSpec
from OCDocker.Workbench.Models import OCScoreInputSpec
from OCDocker.Workbench.OCScoreLayout import ablation_container_paths
from OCDocker.Workbench.OCScoreLayout import build_ocscore_workspace
from OCDocker.Workbench.OCScoreLayout import resolve_ocscore_layout_root
from OCDocker.Workbench.Planner import plan_ocscore_train_command
from OCDocker.Workbench.Preflight import preflight_spec

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Functions
###############################################################################
## Private ##


def _feature_policy_tools() -> dict[str, Any]:
    '''Import FeaturePolicy helpers lazily to avoid a Workbench import cycle.

    Returns
    -------
    dict[str, Any]
        Feature-policy helper callables and constants.
    '''

    from OCDocker.OCScore.Utils.FeaturePolicy import FEATURE_POLICY_METADATA_JSON
    from OCDocker.OCScore.Utils.FeaturePolicy import apply_feature_policy
    from OCDocker.OCScore.Utils.FeaturePolicy import discover_feature_policies
    from OCDocker.OCScore.Utils.FeaturePolicy import discover_candidate_model_features
    from OCDocker.OCScore.Utils.FeaturePolicy import feature_policy_from_mapping
    from OCDocker.OCScore.Utils.FeaturePolicy import feature_policy_to_yaml_text

    return {
        "FEATURE_POLICY_METADATA_JSON": FEATURE_POLICY_METADATA_JSON,
        "apply_feature_policy": apply_feature_policy,
        "discover_feature_policies": discover_feature_policies,
        "discover_candidate_model_features": discover_candidate_model_features,
        "feature_policy_from_mapping": feature_policy_from_mapping,
        "feature_policy_to_yaml_text": feature_policy_to_yaml_text,
    }


def _optional_input_path(body: dict[str, Any], key: str) -> str | None:
    '''Return one trimmed input path from a request body when present.

    Parameters
    ----------
    body : dict[str, Any]
        Parsed JSON request body.
    key : str
        Body field name.

    Returns
    -------
    str or None
        Trimmed path string, or ``None`` when absent.
    '''

    value = str(body.get(key) or "").strip()
    return value or None


def _normalize_feature_source(value: str | None) -> str:
    '''Normalize the requested descriptor source selector.

    Parameters
    ----------
    value : str or None
        Requested source name.

    Returns
    -------
    str
        One of ``auto``, ``pdbbind``, ``dudez``, or ``union``.
    '''

    normalized = str(value or "auto").strip().lower()
    if normalized in {"auto", "pdbbind", "dudez", "union"}:
        return normalized
    raise ValueError("feature_source must be one of: auto, pdbbind, dudez, union.")


_INPUT_PATH_KEYS = ("raw_input_dir", "merged_input", "pdbbind_input", "dudez_input")


def _body_has_input_paths(body: dict[str, Any]) -> bool:
    '''Return whether a request body already specifies modeling input paths.

    Parameters
    ----------
    body : dict[str, Any]
        Parsed JSON request body.

    Returns
    -------
    bool
        ``True`` when at least one supported input path is present.
    '''

    return any(_optional_input_path(body, key) for key in _INPUT_PATH_KEYS)


def _default_workspace_input_paths(root: str | Path) -> dict[str, str]:
    '''Return standard raw_prepare paths relative to the served Workbench root.

    Parameters
    ----------
    root : str or pathlib.Path
        Served OCScore output root.

    Returns
    -------
    dict[str, str]
        ``raw_input_dir``, ``pdbbind_input``, and ``dudez_input`` when present.
    '''

    from OCDocker.OCScore.Utils.RawModelingInput import RAW_DUDEZ_NAME
    from OCDocker.OCScore.Utils.RawModelingInput import RAW_PDBBIND_NAME

    raw_prepare = Path(root).expanduser().resolve() / "raw_prepare"
    pdbbind = raw_prepare / RAW_PDBBIND_NAME
    dudez = raw_prepare / RAW_DUDEZ_NAME
    if not pdbbind.is_file() or not dudez.is_file():
        return {}
    return {"raw_input_dir": str(raw_prepare)}


def _apply_workspace_input_defaults(
    body: dict[str, Any],
    root: str | Path | None,
) -> dict[str, Any]:
    '''Merge workspace-discovered input paths into one request body.

    Parameters
    ----------
    body : dict[str, Any]
        Parsed JSON request body.
    root : str, pathlib.Path, or None
        Served OCScore output root used for auto-discovery.

    Returns
    -------
    dict[str, Any]
        Request body with discovered paths filled in when absent.
    '''

    if _body_has_input_paths(body) or root is None:
        return body

    discovered = _default_workspace_input_paths(root)
    if not discovered:
        return body

    merged = dict(body)
    for key in _INPUT_PATH_KEYS:
        value = discovered.get(key)
        if value:
            merged[key] = value
    merged["_discovered_from"] = discovered["raw_input_dir"]
    return merged


def _load_modeling_table_columns(body: dict[str, Any]) -> tuple[dict[str, Optional[list[str]]], dict[str, str]]:
    '''Read modeling column names from CSV headers referenced by a design request.

    Parameters
    ----------
    body : dict[str, Any]
        Parsed JSON request body with one supported input mode.

    Returns
    -------
    tuple[dict[str, list[str] | None], dict[str, str]]
        Column lists keyed by ``pdbbind`` / ``dudez`` and resolved input paths.

    Raises
    ------
    ValueError
        If no supported input mode was supplied.
    FileNotFoundError
        If referenced input paths do not exist.
    '''

    from OCDocker.OCScore.Utils.RawModelingInput import discover_raw_modeling_input_columns

    return discover_raw_modeling_input_columns(
        raw_input_dir=_optional_input_path(body, "raw_input_dir"),
        merged_input=_optional_input_path(body, "merged_input"),
        pdbbind_input=_optional_input_path(body, "pdbbind_input"),
        dudez_input=_optional_input_path(body, "dudez_input"),
    )


def _table_columns(table: list[str] | Any | None) -> list[str]:
    '''Return column names from a loaded table or pre-read header list.

    Parameters
    ----------
    table : list[str], pandas.DataFrame, or None
        Loaded table or header column list.

    Returns
    -------
    list[str]
        Column names when available.
    '''

    if table is None:
        return []
    if isinstance(table, list):
        return [str(column) for column in table]
    return [str(column) for column in table.columns]


def _select_feature_columns(
    tables: dict[str, Any],
    *,
    feature_source: str,
) -> tuple[str, list[str]]:
    '''Choose which loaded table columns drive candidate-feature discovery.

    Parameters
    ----------
    tables : dict[str, Any]
        Loaded PDBbind and/or DUDEz tables.
    feature_source : str
        One of ``auto``, ``pdbbind``, ``dudez``, or ``union``.

    Returns
    -------
    tuple[str, list[str]]
        Selected source label and ordered column names.

    Raises
    ------
    ValueError
        If the requested source is unavailable.
    '''

    pdbbind_df = tables.get("pdbbind")
    dudez_df = tables.get("dudez")
    if feature_source == "pdbbind":
        if pdbbind_df is None:
            raise ValueError("PDBbind input was not loaded.")
        return "pdbbind", _table_columns(pdbbind_df)
    if feature_source == "dudez":
        if dudez_df is None:
            raise ValueError("DUDEz input was not loaded.")
        return "dudez", _table_columns(dudez_df)
    if feature_source == "union":
        columns: list[str] = []
        for frame in (pdbbind_df, dudez_df):
            for name in _table_columns(frame):
                if name not in columns:
                    columns.append(name)
        if not columns:
            raise ValueError("Union feature discovery requires PDBbind and/or DUDEz tables.")
        return "union", columns
    if pdbbind_df is not None:
        return "pdbbind", _table_columns(pdbbind_df)
    if dudez_df is not None:
        return "dudez", _table_columns(dudez_df)
    raise ValueError("No modeling tables were loaded.")


def _discovery_payload_from_columns(
    columns: list[str],
    *,
    feature_source: str,
    input_paths: dict[str, str],
) -> dict[str, Any]:
    '''Build one candidate-feature discovery payload from raw table columns.

    Parameters
    ----------
    columns : list[str]
        Raw table column names.
    feature_source : str
        Selected descriptor source label.
    input_paths : dict[str, str]
        Resolved input paths used for loading.

    Returns
    -------
    dict[str, Any]
        JSON-safe discovery payload with metadata stripped out of candidates.
    '''

    discovery = _feature_policy_tools()["discover_candidate_model_features"](columns)
    blocks = discovery.blocks
    return {
        "ok": True,
        "feature_source": feature_source,
        "input_paths": input_paths,
        "column_count": len(columns),
        "metadata_columns": discovery.metadata_columns,
        "target_columns": discovery.target_columns,
        "unmatched_columns": discovery.unmatched_columns,
        "candidate_features": discovery.candidate_features,
        "candidate_feature_count": len(discovery.candidate_features),
        "feature_groups": {
            "ligand": list(blocks.ligand),
            "receptor": list(blocks.receptor),
            "scoring": list(blocks.scoring),
            "unmatched": list(blocks.unmatched),
        },
    }


def _resolve_preview_candidate_features(
    body: dict[str, Any],
    layout_root: Path,
) -> tuple[list[str], str | None]:
    '''Resolve candidate features for preview/plan requests.

    Parameters
    ----------
    body : dict[str, Any]
        Parsed JSON request body.
    layout_root : pathlib.Path
        Resolved strict OCScore layout root.

    Returns
    -------
    tuple[list[str], str or None]
        Candidate feature names and a short source label.
    '''

    explicit = body.get("candidate_features")
    if isinstance(explicit, list) and explicit:
        return [str(item) for item in explicit], "request"

    if any(
        _optional_input_path(body, key)
        for key in ("raw_input_dir", "merged_input", "pdbbind_input", "dudez_input")
    ):
        try:
            payload = discover_ablation_input_features(body)
            features = payload.get("candidate_features") or []
            if features:
                source = str(payload.get("feature_source") or "input")
                paths = payload.get("input_paths") or {}
                path_hint = next(iter(paths.values()), source)
                return [str(item) for item in features], f"input:{path_hint}"
        except (OSError, ValueError, FileNotFoundError):
            pass

    features, source = _discover_candidate_features(layout_root)
    if features:
        return features, source
    return [], None


def _build_ocscore_input_spec(body: dict[str, Any]) -> OCScoreInputSpec:
    '''Build one validated OCScore input spec from a design request.

    Parameters
    ----------
    body : dict[str, Any]
        Parsed JSON request body.

    Returns
    -------
    OCScoreInputSpec
        Validated OCScore input selection.

    Raises
    ------
    ValueError
        If no supported input mode was supplied.
    '''

    raw_input_dir = _optional_input_path(body, "raw_input_dir")
    merged_input = _optional_input_path(body, "merged_input")
    pdbbind_input = _optional_input_path(body, "pdbbind_input")
    dudez_input = _optional_input_path(body, "dudez_input")
    if raw_input_dir:
        return OCScoreInputSpec(raw_input_dir=raw_input_dir)
    if merged_input:
        return OCScoreInputSpec(merged_input=merged_input)
    if pdbbind_input and dudez_input:
        return OCScoreInputSpec(pdbbind_input=pdbbind_input, dudez_input=dudez_input)
    raise ValueError(
        "Provide raw_input_dir, merged_input, or both pdbbind_input and dudez_input for planning."
    )


def _policy_catalog_entry(policy: Any) -> dict[str, Any]:
    '''Build one catalog row for a discovered feature policy.

    Parameters
    ----------
    policy : FeaturePolicy
        Loaded feature policy.

    Returns
    -------
    dict[str, Any]
        JSON-safe catalog row.
    '''

    return {
        "name": policy.name,
        "description": policy.description,
        "source_kind": policy.source_kind,
        "source_path": str(policy.source_path),
        "request": policy.request_payload(),
    }


def _discover_candidate_features(layout_root: Path) -> tuple[list[str], str | None]:
    '''Load candidate model features from one replica metadata file when present.

    Parameters
    ----------
    layout_root : pathlib.Path
        Resolved strict OCScore layout root.

    Returns
    -------
    tuple[list[str], str or None]
        Candidate feature names and the metadata path they came from.
    '''

    tools = _feature_policy_tools()
    metadata_name = tools["FEATURE_POLICY_METADATA_JSON"]
    search_roots: list[Path] = []
    if layout_root.is_dir():
        search_roots.append(layout_root)
    for container in ablation_container_paths(layout_root):
        for study_path in sorted(path for path in container.iterdir() if path.is_dir()):
            search_roots.append(study_path)

    for study_root in search_roots:
        replica_dirs = sorted(
            path for path in study_root.iterdir() if path.is_dir() and path.name.startswith("replica")
        )
        if not replica_dirs:
            continue
        metadata_path = replica_dirs[0] / metadata_name
        if not metadata_path.is_file():
            continue
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        candidates = payload.get("candidate_features_before_policy")
        if isinstance(candidates, list) and candidates:
            return [str(item) for item in candidates], str(metadata_path)
    return [], None


def _preview_payload(policy_data: dict[str, Any], candidate_features: list[str]) -> dict[str, Any]:
    '''Build one feature-policy preview payload.

    Parameters
    ----------
    policy_data : dict[str, Any]
        Draft feature-policy mapping.
    candidate_features : list[str]
        Candidate model feature names.

    Returns
    -------
    dict[str, Any]
        JSON-safe preview payload.
    '''

    tools = _feature_policy_tools()
    feature_policy_from_mapping = tools["feature_policy_from_mapping"]
    feature_policy_to_yaml_text = tools["feature_policy_to_yaml_text"]
    apply_feature_policy = tools["apply_feature_policy"]
    policy = feature_policy_from_mapping(policy_data)
    if not candidate_features:
        return {
            "ok": True,
            "policy_name": policy.name,
            "candidate_source": None,
            "candidate_feature_count": 0,
            "preview_available": False,
            "message": (
                "No candidate feature list was found in this workspace. "
                "Save the policy YAML and run training to evaluate it."
            ),
            "policy_yaml": feature_policy_to_yaml_text(policy_data),
        }

    application = apply_feature_policy(policy, candidate_features)
    metadata = application.to_metadata()
    final_features = metadata.get("final_candidate_features_before_reduction") or []
    excluded = metadata.get("excluded_features_found") or []
    return {
        "ok": True,
        "policy_name": policy.name,
        "candidate_source": None,
        "candidate_feature_count": len(candidate_features),
        "preview_available": True,
        "kept_feature_count": len(final_features),
        "excluded_feature_count": len(excluded),
        "missing_exclude_features": metadata.get("missing_exclude_features") or [],
        "patterns_with_no_matches": metadata.get("patterns_with_no_matches") or [],
        "kept_features_sample": list(final_features[:24]),
        "excluded_features_sample": list(excluded[:24]),
        "policy_yaml": feature_policy_to_yaml_text(policy_data),
    }


def _resolve_ablation_container(layout_root: Path) -> Path:
    '''Return the preferred ablation output container for one layout root.

    Parameters
    ----------
    layout_root : pathlib.Path
        Resolved strict OCScore layout root.

    Returns
    -------
    pathlib.Path
        Existing ablation container or a default ``ablations/`` path.
    '''

    containers = list(ablation_container_paths(layout_root))
    if containers:
        return containers[0]
    return layout_root / "ablations"


def _build_design_context(root: Path) -> dict[str, Any]:
    '''Build catalog and workspace defaults for the ablation designer UI.

    Parameters
    ----------
    root : pathlib.Path
        Served OCScore output root.

    Returns
    -------
    dict[str, Any]
        JSON-safe design context payload.
    '''

    layout_root = resolve_ocscore_layout_root(root)
    workspace = build_ocscore_workspace(root)
    protocol = workspace.protocol
    candidate_features, candidate_source = _discover_candidate_features(layout_root)
    discovery = _feature_policy_tools()["discover_feature_policies"]()
    catalog = [_policy_catalog_entry(discovery.policies[name]) for name in discovery.available_names]
    existing_names = sorted({study.study_name for study in workspace.ablation_studies})
    ablation_container = _resolve_ablation_container(layout_root)
    protocol_path = str(protocol.source_path) if protocol and protocol.source_path else ""
    discovered = _default_workspace_input_paths(root)
    discovered_inputs = {"ok": bool(discovered)}
    if discovered:
        raw_prepare = Path(discovered["raw_input_dir"])
        discovered_inputs.update(
            {
                "discovered_from": discovered["raw_input_dir"],
                "raw_input_dir": discovered["raw_input_dir"],
                "pdbbind_input": str(raw_prepare / "raw_pdbbind.csv"),
                "dudez_input": str(raw_prepare / "raw_dudez.csv"),
            }
        )
    return {
        "ok": True,
        "read_only": True,
        "workspace_root": str(root),
        "layout_root": str(layout_root),
        "protocol_name": protocol.protocol_name if protocol else "",
        "protocol_path": protocol_path,
        "ablation_container": str(ablation_container),
        "existing_ablation_names": existing_names,
        "candidate_features": candidate_features,
        "candidate_source": candidate_source,
        "discovered_inputs": discovered_inputs,
        "catalog": catalog,
    }


def _coerce_policy_draft(body: dict[str, Any]) -> dict[str, Any]:
    '''Extract one draft feature-policy mapping from an API request body.

    Parameters
    ----------
    body : dict[str, Any]
        Parsed JSON request body.

    Returns
    -------
    dict[str, Any]
        Draft feature-policy mapping.

    Raises
    ------
    ValueError
        If the request body does not contain a policy object.
    '''

    policy = body.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("Request body must include a 'policy' object.")
    return policy


def _build_ablation_spec(body: dict[str, Any], *, layout_root: Path) -> OCScoreAblationSpec:
    '''Build one validated OCScore ablation spec from a design request.

    Parameters
    ----------
    body : dict[str, Any]
        Parsed JSON request body.
    layout_root : pathlib.Path
        Resolved strict OCScore layout root.

    Returns
    -------
    OCScoreAblationSpec
        Validated ablation spec.

    Raises
    ------
    ValueError
        If required request fields are missing.
    '''

    policy_data = _coerce_policy_draft(body)
    policy = _feature_policy_tools()["feature_policy_from_mapping"](policy_data)
    policy_name = policy.name

    protocol = str(body.get("protocol") or "").strip()
    if not protocol:
        raise ValueError("Request body must include 'protocol'.")

    raw_input_dir = _optional_input_path(body, "raw_input_dir")
    if not _body_has_input_paths(body):
        raise ValueError(
            "Provide raw_input_dir, merged_input, or both pdbbind_input and dudez_input, "
            "or serve a Workbench root that contains raw_prepare/raw_pdbbind.csv and raw_dudez.csv."
        )

    output_dir = str(body.get("output_dir") or "").strip()
    if not output_dir:
        output_dir = str(_resolve_ablation_container(layout_root) / policy_name)

    policy_yml_path = str(body.get("policy_yml_path") or "").strip()
    if not policy_yml_path:
        policy_yml_path = str(Path(body.get("policy_dir") or layout_root / "Ablations") / f"{policy_name}.yml")

    campaign_name = str(body.get("name") or f"ablation-{policy_name}").strip()
    description = str(body.get("description") or policy.description or "").strip()

    return OCScoreAblationSpec(
        name=campaign_name,
        protocol=protocol,
        inputs=_build_ocscore_input_spec(body),
        output_dir=output_dir,
        feature_policies=FeaturePolicySelection(
            names=(policy_name,),
            policy_ymls=(policy_yml_path,),
        ),
        include_full_reference=False,
        description=description,
        tags=("workbench", "ocscore", "ablation", "designed"),
    )


## Public ##


def build_ablation_design_context(root: str | Path) -> dict[str, Any]:
    '''Build catalog and workspace defaults for the ablation designer UI.

    Parameters
    ----------
    root : str or pathlib.Path
        Served OCScore output root.

    Returns
    -------
    dict[str, Any]
        JSON-safe design context payload.
    '''

    return _build_design_context(Path(root).expanduser().resolve())


def discover_ablation_input_features(
    body: dict[str, Any],
    root: str | Path | None = None,
) -> dict[str, Any]:
    '''Discover candidate model features from raw PDBbind/DUDEz modeling inputs.

    Parameters
    ----------
    body : dict[str, Any]
        Parsed JSON request body with input paths and optional ``feature_source``.
    root : str, pathlib.Path, or None
        Served OCScore output root used to auto-discover ``raw_prepare/`` tables
        when explicit input paths are omitted.

    Returns
    -------
    dict[str, Any]
        JSON-safe payload with metadata/target columns removed from candidates.

    Raises
    ------
    ValueError
        If input paths or ``feature_source`` are invalid.
    FileNotFoundError
        If referenced input paths do not exist.
    '''

    body = _apply_workspace_input_defaults(body, root)
    if not _body_has_input_paths(body):
        raise ValueError(
            "No raw modeling input files found. Expected "
            f"{Path(root).expanduser().resolve() / 'raw_prepare' / 'raw_pdbbind.csv'} and "
            "raw_dudez.csv under the served Workbench root, or set input paths in Run settings."
        )

    feature_source = _normalize_feature_source(body.get("feature_source"))
    tables, input_paths = _load_modeling_table_columns(body)
    selected_source, columns = _select_feature_columns(tables, feature_source=feature_source)
    payload = _discovery_payload_from_columns(
        columns,
        feature_source=selected_source,
        input_paths=input_paths,
    )
    payload["columns_only"] = True
    discovered_from = body.get("_discovered_from")
    if discovered_from:
        payload["discovered_from"] = discovered_from
        payload["auto_discovered"] = True
    resolved_inputs = {
        key: value
        for key in _INPUT_PATH_KEYS
        if (value := _optional_input_path(body, key))
    }
    if resolved_inputs:
        payload["resolved_inputs"] = resolved_inputs
    return payload


def preview_ablation_design(root: str | Path, body: dict[str, Any]) -> dict[str, Any]:
    '''Preview one draft feature policy against workspace candidate features.

    Parameters
    ----------
    root : str or pathlib.Path
        Served OCScore output root.
    body : dict[str, Any]
        Parsed JSON request body containing a ``policy`` mapping.

    Returns
    -------
    dict[str, Any]
        JSON-safe preview payload.

    Raises
    ------
    ValueError
        If the request body is invalid.
    '''

    layout_root = resolve_ocscore_layout_root(root)
    body = _apply_workspace_input_defaults(body, root)
    policy_data = _coerce_policy_draft(body)
    candidates, candidate_source = _resolve_preview_candidate_features(body, layout_root)

    payload = _preview_payload(policy_data, candidates)
    payload["candidate_source"] = candidate_source
    return payload


def plan_ablation_design(root: str | Path, body: dict[str, Any]) -> dict[str, Any]:
    '''Validate a draft ablation, emit YAML, command plan, and preflight checks.

    Parameters
    ----------
    root : str or pathlib.Path
        Served OCScore output root.
    body : dict[str, Any]
        Parsed JSON request body containing policy and run settings.

    Returns
    -------
    dict[str, Any]
        JSON-safe plan payload with YAML, command, spec, and preflight report.

    Raises
    ------
    ValueError
        If the request body is invalid.
    '''

    root_path = Path(root).expanduser().resolve()
    layout_root = resolve_ocscore_layout_root(root_path)
    body = _apply_workspace_input_defaults(body, root_path)
    policy_data = _coerce_policy_draft(body)
    spec = _build_ablation_spec(body, layout_root=layout_root)
    plan = plan_ocscore_train_command(spec)
    report = preflight_spec(spec, spec_path=root_path)
    policy_yml_path = str(spec.feature_policies.policy_ymls[0]) if spec.feature_policies.policy_ymls else ""
    feature_policy_to_yaml_text = _feature_policy_tools()["feature_policy_to_yaml_text"]
    return {
        "ok": True,
        "read_only": True,
        "policy_name": policy_data.get("name"),
        "policy_yaml": feature_policy_to_yaml_text(policy_data),
        "policy_yml_path": policy_yml_path,
        "spec": model_to_data(spec),
        "planned_command": " ".join(plan.command),
        "planned_command_argv": list(plan.command),
        "preflight": model_to_data(report),
    }


def write_ablation_design_policy(root: str | Path, body: dict[str, Any]) -> dict[str, Any]:
    '''Write one draft feature-policy YAML into the served workspace layout.

    Parameters
    ----------
    root : str or pathlib.Path
        Served OCScore output root.
    body : dict[str, Any]
        Parsed JSON request body with ``policy``, ``policy_yml_path``,
        ``confirm``, and optional ``overwrite``.

    Returns
    -------
    dict[str, Any]
        JSON-safe write result.

    Raises
    ------
    ValueError
        If confirmation, paths, or overwrite guards fail.
    FileExistsError
        If the target file exists and ``overwrite`` is not set.
    '''

    if not body.get("confirm"):
        raise ValueError("Set confirm: true to write a policy YAML into the workspace.")

    root_path = Path(root).expanduser().resolve()
    layout_root = resolve_ocscore_layout_root(root_path)
    policy_data = _coerce_policy_draft(body)
    tools = _feature_policy_tools()
    feature_policy_to_yaml_text = tools["feature_policy_to_yaml_text"]
    yaml_text = str(body.get("policy_yaml") or "").strip()
    if not yaml_text:
        yaml_text = feature_policy_to_yaml_text(policy_data)

    policy_yml_path = str(body.get("policy_yml_path") or "").strip()
    if not policy_yml_path:
        policy_name = str(policy_data.get("name") or "").strip()
        if not policy_name:
            raise ValueError("Request body must include policy_yml_path or a policy name.")
        policy_yml_path = str(Path("Ablations") / f"{policy_name}.yml")

    target = Path(policy_yml_path).expanduser()
    if not target.is_absolute():
        target = (layout_root / target).resolve()
    else:
        target = target.resolve()

    layout_resolved = layout_root.resolve()
    try:
        target.relative_to(layout_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Policy path must stay inside the workspace layout root ({layout_resolved})."
        ) from exc

    from OCDocker.OCScore.Utils.FeaturePolicy import BUNDLED_FEATURE_POLICY_DIR

    bundled_resolved = BUNDLED_FEATURE_POLICY_DIR.resolve()
    try:
        target.relative_to(bundled_resolved)
    except ValueError:
        pass
    else:
        raise ValueError("Bundled/shipped feature policies cannot be overwritten from the Workbench.")

    if target.is_file() and not body.get("overwrite"):
        raise FileExistsError(f"Policy file already exists: {target}. Set overwrite: true to replace it.")

    discovery = tools["discover_feature_policies"]()
    bundled_names = {
        name
        for name, policy in discovery.policies.items()
        if policy.source_kind == "bundled"
    }
    policy_name = str(policy_data.get("name") or "").strip()
    if policy_name in bundled_names and target.name == f"{policy_name}.yml":
        bundled_path = bundled_resolved / f"{policy_name}.yml"
        if bundled_path.is_file() and target.resolve() == bundled_path.resolve():
            raise ValueError(
                f'Bundled policy "{policy_name}" is read-only. Choose a different name or path.'
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml_text if yaml_text.endswith("\n") else f"{yaml_text}\n", encoding="utf-8")
    return {
        "ok": True,
        "read_only": False,
        "written_path": str(target),
        "policy_name": policy_name or target.stem,
        "policy_yaml": yaml_text,
    }


def handle_ablation_design_post(root: str | Path, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
    '''Dispatch one ablation-design POST endpoint.

    Parameters
    ----------
    root : str or pathlib.Path
        Served OCScore output root.
    endpoint : str
        Request path.
    body : dict[str, Any]
        Parsed JSON request body.

    Returns
    -------
    dict[str, Any]
        JSON-safe response payload.

    Raises
    ------
    ValueError
        If the endpoint is unknown or the request body is invalid.
    '''

    path = endpoint.rstrip("/")
    if path == "/api/ablation-design/features":
        return discover_ablation_input_features(body, root)
    if path == "/api/ablation-design/preview":
        return preview_ablation_design(root, body)
    if path == "/api/ablation-design/plan":
        return plan_ablation_design(root, body)
    if path == "/api/ablation-design/write":
        return write_ablation_design_policy(root, body)
    raise ValueError(f"Unknown ablation design endpoint: {endpoint}")


__all__ = [
    "build_ablation_design_context",
    "discover_ablation_input_features",
    "handle_ablation_design_post",
    "plan_ablation_design",
    "preview_ablation_design",
    "write_ablation_design_policy",
]
