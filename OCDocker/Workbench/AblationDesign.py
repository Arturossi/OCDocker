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
    from OCDocker.OCScore.Utils.FeaturePolicy import feature_policy_from_mapping
    from OCDocker.OCScore.Utils.FeaturePolicy import feature_policy_to_yaml_text

    return {
        "FEATURE_POLICY_METADATA_JSON": FEATURE_POLICY_METADATA_JSON,
        "apply_feature_policy": apply_feature_policy,
        "discover_feature_policies": discover_feature_policies,
        "feature_policy_from_mapping": feature_policy_from_mapping,
        "feature_policy_to_yaml_text": feature_policy_to_yaml_text,
    }


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
    '''Load candidate model features from workspace replica metadata when present.

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
        for replica_dir in replica_dirs:
            metadata_path = replica_dir / metadata_name
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

    raw_input_dir = str(body.get("raw_input_dir") or "").strip()
    if not raw_input_dir:
        raise ValueError("Request body must include 'raw_input_dir'.")

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
        inputs=OCScoreInputSpec(raw_input_dir=raw_input_dir),
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
    policy_data = _coerce_policy_draft(body)
    candidate_features = body.get("candidate_features")
    candidate_source = None
    if isinstance(candidate_features, list) and candidate_features:
        candidates = [str(item) for item in candidate_features]
        candidate_source = "request"
    else:
        candidates, candidate_source = _discover_candidate_features(layout_root)

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
    if path == "/api/ablation-design/preview":
        return preview_ablation_design(root, body)
    if path == "/api/ablation-design/plan":
        return plan_ablation_design(root, body)
    raise ValueError(f"Unknown ablation design endpoint: {endpoint}")


__all__ = [
    "build_ablation_design_context",
    "handle_ablation_design_post",
    "plan_ablation_design",
    "preview_ablation_design",
]
