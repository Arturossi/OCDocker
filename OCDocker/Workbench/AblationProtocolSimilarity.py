#!/usr/bin/env python3

# Description
###############################################################################
'''
Expanded feature-policy similarity for OCScore ablation protocols.

Each executed study uses ``feature_policy_metadata.json`` from one replica when
available. Catalog-only policies fall back to bundled YAML policies resolved with
``apply_feature_policy`` against the same workspace candidate discovery path used
by :mod:`OCDocker.Workbench.AblationDesign`.

The read-only payload powers ``GET /api/ablation-protocol-similarity`` and the
Protocol similarity zone in the Ablation dashboard tab.
'''

# Imports
###############################################################################
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

from OCDocker.Workbench.Ablation import parse_ablation_metric
from OCDocker.Workbench.AblationDesign import _discover_candidate_features as discover_workspace_candidate_features
from OCDocker.Workbench.AblationDesign import discover_ablation_input_features
from OCDocker.Workbench.Models import InventoryIssue
from OCDocker.Workbench.Models import WorkbenchAblationProtocolClusterSummary
from OCDocker.Workbench.Models import WorkbenchAblationProtocolFamilyState
from OCDocker.Workbench.Models import WorkbenchAblationProtocolReferenceDiff
from OCDocker.Workbench.Models import WorkbenchAblationProtocolSimilarity
from OCDocker.Workbench.Models import WorkbenchAblationProtocolSimilarityEntry
from OCDocker.Workbench.OCScoreLayout import ablation_container_paths
from OCDocker.Workbench.OCScoreLayout import resolve_ocscore_layout_root

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Constants
###############################################################################

DEFAULT_REFERENCE_POLICY = "full_ocscore"
COARSE_FAMILY_IDS = ("ligand", "receptor", "scoring", "unmatched")
_FEATURE_POLICY_TOOLS: dict[str, Any] | None = None
_STUDY_PROTOCOL_JSON = ("staged_optuna_protocol.json", "replicas_protocol.json")

# Classes
###############################################################################


@dataclass(frozen=True)
class _FamilyDefinition:
    """One coarse or prefix feature family and its member columns."""

    family_id: str
    members: frozenset[str]


@dataclass(frozen=True)
class _ResolvedProtocol:
    """One policy resolved to an expanded feature set and family states."""

    policy_name: str
    policy_description: str
    policy_source_kind: str
    policy_source_path: Path
    expanded_features: frozenset[str]
    family_states: tuple[WorkbenchAblationProtocolFamilyState, ...]


# Functions
###############################################################################
## Private ##


def _feature_policy_tools() -> dict[str, Any]:
    '''Return lazily imported FeaturePolicy helpers.

    Returns
    -------
    dict[str, Any]
        Mapping of helper callables and constants used by this module.
    '''
    global _FEATURE_POLICY_TOOLS
    if _FEATURE_POLICY_TOOLS is not None:
        return _FEATURE_POLICY_TOOLS
    from OCDocker.OCScore.Utils.FeaturePolicy import FEATURE_POLICY_METADATA_JSON
    from OCDocker.OCScore.Utils.FeaturePolicy import apply_feature_policy
    from OCDocker.OCScore.Utils.FeaturePolicy import discover_candidate_model_features
    from OCDocker.OCScore.Utils.FeaturePolicy import discover_feature_policies
    from OCDocker.OCScore.Utils.FeaturePolicy import feature_policy_from_mapping

    _FEATURE_POLICY_TOOLS = {
        "FEATURE_POLICY_METADATA_JSON": FEATURE_POLICY_METADATA_JSON,
        "apply_feature_policy": apply_feature_policy,
        "discover_candidate_model_features": discover_candidate_model_features,
        "discover_feature_policies": discover_feature_policies,
        "feature_policy_from_mapping": feature_policy_from_mapping,
    }
    return _FEATURE_POLICY_TOOLS


def _first_replica_dir(study_root: Path) -> Path | None:
    '''Return the lexicographically first replica directory under one study root.

    Parameters
    ----------
    study_root : Path
        Baseline or ablation study directory.

    Returns
    -------
    Path or None
        First ``replica_*`` directory when present.
    '''
    if not study_root.is_dir():
        return None
    best: Path | None = None
    for path in study_root.iterdir():
        if not path.is_dir() or not path.name.startswith("replica"):
            continue
        if best is None or path.name < best.name:
            best = path
    return best


def _iter_study_file_paths(study_root: Path, filename: str) -> tuple[Path, ...]:
    '''Yield a study-level file path, then the same file under the first replica.

    Parameters
    ----------
    study_root : Path
        Baseline or ablation study directory.
    filename : str
        File name to resolve.

    Returns
    -------
    tuple[Path, ...]
        Existing paths in study-first order.
    '''
    paths: list[Path] = []
    study_candidate = study_root / filename
    if study_candidate.is_file():
        paths.append(study_candidate)
        return tuple(paths)
    replica_dir = _first_replica_dir(study_root)
    if replica_dir is not None:
        replica_candidate = replica_dir / filename
        if replica_candidate.is_file():
            paths.append(replica_candidate)
    return tuple(paths)


def _unique_preserve_order(values: Sequence[str]) -> list[str]:
    '''Return unique strings in first-seen order.

    Parameters
    ----------
    values : Sequence[str]
        Input strings, possibly with duplicates.

    Returns
    -------
    list[str]
        De-duplicated values preserving original order.
    '''
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _resolve_workspace_candidate_features(
    root: Path,
    layout_root: Path,
    study_paths: Mapping[str, Path],
) -> tuple[list[str], str | None]:
    '''Resolve the shared candidate universe using Workbench ablation-design rules.

    Parameters
    ----------
    root : Path
        Served Workbench root.
    layout_root : Path
        Resolved OCScore layout root.
    study_paths : Mapping[str, Path]
        Workspace study folders keyed by policy or study name.

    Returns
    -------
    tuple[list[str], str | None]
        Candidate columns and discovery source path or label.
    '''
    candidates, source = discover_workspace_candidate_features(layout_root)
    union = set(candidates)
    union.update(_union_study_candidate_features(study_paths))
    if union:
        return sorted(union), source or "feature_policy_metadata.json"
    try:
        payload = discover_ablation_input_features({}, root=root)
    except (ValueError, FileNotFoundError, OSError):
        return [], None
    discovered = payload.get("candidate_features")
    if not isinstance(discovered, list) or not discovered:
        return [], None
    discovered_from = payload.get("discovered_from") or payload.get("input_paths", {}).get("pdbbind_input")
    return [str(item) for item in discovered], str(discovered_from) if discovered_from else None


def _union_study_candidate_features(study_paths: Mapping[str, Path]) -> set[str]:
    '''Union candidate columns recorded across workspace study metadata files.

    Parameters
    ----------
    study_paths : Mapping[str, Path]
        Workspace study folders keyed by policy or study name.

    Returns
    -------
    set[str]
        Candidate model feature names inferred from study metadata.
    '''
    merged: set[str] = set()
    for study_path in study_paths.values():
        loaded = _load_study_feature_policy_metadata(study_path)
        if loaded is None:
            continue
        metadata, _ = loaded
        merged.update(_candidate_features_from_metadata(metadata))
        merged.update(_expanded_features_from_metadata(metadata))
        for key in ("included_features_found", "excluded_features_found"):
            values = metadata.get(key)
            if isinstance(values, list):
                merged.update(str(item) for item in values)
    return merged


def _study_paths_by_policy(layout_root: Path) -> dict[str, Path]:
    '''Map bundled policy names and study folder names to study roots.

    Parameters
    ----------
    layout_root : Path
        Resolved OCScore layout root.

    Returns
    -------
    dict[str, Path]
        Lookup from policy or study name to study directory.
    '''
    paths: dict[str, Path] = {}
    if _layout_has_baseline_replicas(layout_root):
        paths["baseline"] = layout_root
        paths[DEFAULT_REFERENCE_POLICY] = layout_root
    for container in ablation_container_paths(layout_root):
        if not container.is_dir():
            continue
        for study_path in sorted(path for path in container.iterdir() if path.is_dir()):
            if study_path.name.startswith("."):
                continue
            paths[study_path.name] = study_path
    return paths


def _study_is_fully_run(study_path: Path, *, layout_root: Path) -> bool:
    '''Return whether every expected replica in a study finished PDBbind and DUDEz.

    Parameters
    ----------
    study_path : Path
        Baseline or ablation study directory.
    layout_root : Path
        Resolved OCScore layout root used to infer expected replica count.

    Returns
    -------
    bool
        True when all expected replica slots exist and completed the pipeline.
    '''
    from OCDocker.Workbench.OCScoreLayout import _collect_replica_paths
    from OCDocker.Workbench.OCScoreLayout import _infer_study_replica_count
    from OCDocker.Workbench.OCScoreLayout import _replica_pipeline_complete

    if not study_path.is_dir():
        return False
    replica_paths = _collect_replica_paths(study_path)
    if not replica_paths:
        return False
    expected = _infer_study_replica_count(study_path, layout_root=layout_root)
    if expected < 1:
        return False
    completed = 0
    for slot in range(1, expected + 1):
        replica_path = replica_paths.get(slot)
        if replica_path is None or not _replica_pipeline_complete(replica_path):
            return False
        completed += 1
    return completed >= expected


def _completed_run_id(study_path: Path | None, *, layout_root: Path, policy_name: str) -> str | None:
    '''Map a study folder to a run id only when the study fully completed.

    Parameters
    ----------
    study_path : Path or None
        Study directory when present in the workspace layout.
    layout_root : Path
        Resolved OCScore layout root.
    policy_name : str
        Policy or study name used as the run identifier.

    Returns
    -------
    str or None
        ``policy_name`` when fully run, otherwise ``None``.
    '''
    if study_path is None:
        return None
    if _study_is_fully_run(study_path, layout_root=layout_root):
        return policy_name
    return None


def _load_study_feature_policy_metadata(study_path: Path) -> tuple[Mapping[str, Any], Path] | None:
    '''Load the best ``feature_policy_metadata.json`` available for one study.

    Scans every replica directory and prefers metadata with an expanded feature
    list, then explicit candidate columns, then any recorded policy request.

    Parameters
    ----------
    study_path : Path
        Baseline or ablation study directory.

    Returns
    -------
    tuple[Mapping[str, Any], Path] or None
        Parsed metadata payload and file path when present.
    '''
    metadata_name = _feature_policy_tools()["FEATURE_POLICY_METADATA_JSON"]
    best_payload: Mapping[str, Any] | None = None
    best_path: Path | None = None
    best_score = -1
    search_roots = [study_path]
    search_roots.extend(
        sorted(path for path in study_path.iterdir() if path.is_dir() and path.name.startswith("replica"))
    )
    for root in search_roots:
        metadata_path = root / metadata_name
        if not metadata_path.is_file():
            continue
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        score = 0
        if _expanded_features_from_metadata(payload):
            score += 100
        if _candidate_features_from_metadata(payload):
            score += 10
        if payload.get("included_patterns_requested") or payload.get("included_features_requested"):
            score += 1
        if score > best_score:
            best_payload = payload
            best_path = metadata_path
            best_score = score
    if best_payload is not None and best_path is not None:
        return best_payload, best_path
    return None


def _policy_from_metadata(metadata: Mapping[str, Any], fallback_name: str, discovery: Any) -> Any | None:
    '''Resolve one feature policy from OCDocker metadata or the bundled catalog.

    Parameters
    ----------
    metadata : Mapping[str, Any]
        Parsed ``feature_policy_metadata.json`` content.
    fallback_name : str
        Study or bundled policy name used when metadata omits ``feature_policy_name``.
    discovery : Any
        Feature policy discovery payload.

    Returns
    -------
    Any or None
        Feature policy object when metadata or the catalog defines one.
    '''
    policy_name = str(metadata.get("feature_policy_name") or fallback_name)
    if policy_name in discovery.policies:
        return discovery.policies[policy_name]
    if fallback_name in discovery.policies:
        return discovery.policies[fallback_name]

    include_features = metadata.get("included_features_requested", metadata.get("include_features", ()))
    include_patterns = metadata.get("included_patterns_requested", metadata.get("include_patterns", ()))
    exclude_features = metadata.get("excluded_features_requested", metadata.get("exclude_features", ()))
    exclude_patterns = metadata.get("excluded_patterns_requested", metadata.get("exclude_patterns", ()))
    if not any((include_features, include_patterns, exclude_features, exclude_patterns)):
        return None

    source_kind = str(metadata.get("feature_policy_source_kind") or "bundled")
    if source_kind not in {"bundled", "user_dir", "explicit_yml", "draft"}:
        source_kind = "bundled"
    source_path = metadata.get("feature_policy_source_path")
    try:
        return _feature_policy_tools()["feature_policy_from_mapping"](
            {
                "name": policy_name,
                "description": metadata.get("feature_policy_description", ""),
                "include_features": include_features,
                "include_patterns": include_patterns,
                "exclude_features": exclude_features,
                "exclude_patterns": exclude_patterns,
                "allow_missing_exclude_features": metadata.get("allow_missing_exclude_features", True),
                "allow_empty_policy": metadata.get("allow_empty_policy", False),
            },
            source_kind=source_kind,
            source_path=source_path,
        )
    except ValueError:
        return None


def _candidate_features_from_metadata(metadata: Mapping[str, Any]) -> list[str]:
    '''Return candidate columns recorded in one feature-policy metadata payload.

    Parameters
    ----------
    metadata : Mapping[str, Any]
        Parsed ``feature_policy_metadata.json`` content.

    Returns
    -------
    list[str]
        Candidate model feature names when recorded, else an empty list.
    '''
    candidates = metadata.get("candidate_features_before_policy")
    if isinstance(candidates, list) and candidates:
        return [str(item) for item in candidates]
    return []


def _expanded_features_from_metadata(metadata: Mapping[str, Any]) -> frozenset[str]:
    '''Return the expanded feature set recorded by OCDocker training.

    Parameters
    ----------
    metadata : Mapping[str, Any]
        Parsed ``feature_policy_metadata.json`` content.

    Returns
    -------
    frozenset[str]
        Final candidate features before train-only reduction.
    '''
    final_features = metadata.get("final_candidate_features_before_reduction")
    if not isinstance(final_features, list) or not final_features:
        return frozenset()
    return frozenset(str(item) for item in final_features)


def _resolved_protocol_from_metadata(
    name: str,
    metadata: Mapping[str, Any],
    metadata_path: Path,
    definitions: Sequence[_FamilyDefinition],
) -> _ResolvedProtocol | None:
    '''Build one resolved protocol from OCDocker feature-policy metadata.

    Parameters
    ----------
    name : str
        Selected policy or study name for display.
    metadata : Mapping[str, Any]
        Parsed ``feature_policy_metadata.json`` content.
    metadata_path : Path
        Metadata file path used for provenance.
    definitions : Sequence[_FamilyDefinition]
        Family definitions for rollup states.

    Returns
    -------
    _ResolvedProtocol or None
        Resolved protocol when metadata contains an expanded feature set.
    '''
    expanded = _expanded_features_from_metadata(metadata)
    if expanded:
        source_path = metadata.get("feature_policy_source_path")
        return _ResolvedProtocol(
            policy_name=name,
            policy_description=str(metadata.get("feature_policy_description") or ""),
            policy_source_kind=str(metadata.get("feature_policy_source_kind") or "bundled"),
            policy_source_path=Path(source_path) if source_path else metadata_path,
            expanded_features=expanded,
            family_states=_family_states(expanded, definitions),
        )
    return _resolved_protocol_from_included_features(name, metadata, metadata_path, definitions)


def _resolved_protocol_from_included_features(
    name: str,
    metadata: Mapping[str, Any],
    metadata_path: Path,
    definitions: Sequence[_FamilyDefinition],
) -> _ResolvedProtocol | None:
    '''Build one resolved protocol from OCDocker ``included_features_found`` metadata.'''
    included = metadata.get("included_features_found")
    if not isinstance(included, list) or not included:
        return None
    expanded = frozenset(str(item) for item in included)
    source_path = metadata.get("feature_policy_source_path")
    return _ResolvedProtocol(
        policy_name=name,
        policy_description=str(metadata.get("feature_policy_description") or ""),
        policy_source_kind=str(metadata.get("feature_policy_source_kind") or "bundled"),
        policy_source_path=Path(source_path) if source_path else metadata_path,
        expanded_features=expanded,
        family_states=_family_states(expanded, definitions),
    )


def _workspace_policy_dirs(layout_root: Path) -> tuple[Path, ...]:
    '''Collect workspace directories that may contain custom feature policies.

    Parameters
    ----------
    layout_root : Path
        OCScore layout root.

    Returns
    -------
    tuple[pathlib.Path, ...]
        De-duplicated policy search directories under the layout.
    '''
    dirs: list[Path] = []
    seen: set[str] = set()
    for candidate in (
        layout_root / "Ablations",
        layout_root / "ablations",
    ):
        key = str(candidate.resolve())
        if candidate.is_dir() and key not in seen:
            seen.add(key)
            dirs.append(candidate)
    for container in ablation_container_paths(layout_root):
        for name in ("Ablations", "Policies"):
            candidate = container / name
            key = str(candidate.resolve())
            if candidate.is_dir() and key not in seen:
                seen.add(key)
                dirs.append(candidate)
    return tuple(dirs)


def _executed_policy_names(layout_root: Path, rows: Mapping[str, Any]) -> set[str]:
    '''Collect policy names with workspace output folders or ablation result manifests.

    Parameters
    ----------
    layout_root : Path
        OCScore layout root scanned for ``ablation/`` and ``ablations/`` study folders.
    rows : Mapping[str, Any]
        Ablation rows keyed by run id from :func:`OCDocker.Workbench.Ablation._load_ablation_rows`.

    Returns
    -------
    set[str]
        Policy names considered executed in the workspace.
    '''
    names = {row.policy_name for row in rows.values() if getattr(row, "is_ablation", False)}
    for container in ablation_container_paths(layout_root):
        if not container.is_dir():
            continue
        for study_path in sorted(path for path in container.iterdir() if path.is_dir()):
            if study_path.name.startswith("."):
                continue
            names.add(study_path.name)
    return names


def _layout_has_baseline_replicas(layout_root: Path) -> bool:
    '''Return whether the layout root contains at least one baseline replica folder.

    Parameters
    ----------
    layout_root : Path
        Resolved OCScore layout root.

    Returns
    -------
    bool
        True when a ``replica_*`` directory exists directly under the layout root.
    '''
    if not layout_root.is_dir():
        return False
    return any(path.is_dir() and path.name.startswith("replica") for path in layout_root.iterdir())


def _fast_layout_protocol_names(layout_root: Path, discovery: Any) -> tuple[str, ...]:
    '''List workspace protocol names from layout folders without a full workspace scan.

    Parameters
    ----------
    layout_root : Path
        Resolved OCScore layout root.
    discovery : Any
        Feature policy discovery payload.

    Returns
    -------
    tuple[str, ...]
        Baseline reference policy and ablation study folder names.
    '''
    names: list[str] = []
    if _layout_has_baseline_replicas(layout_root):
        if DEFAULT_REFERENCE_POLICY in discovery.policies:
            names.append(DEFAULT_REFERENCE_POLICY)
        else:
            names.append("baseline")
    for container in ablation_container_paths(layout_root):
        if not container.is_dir():
            continue
        for study_path in sorted(path for path in container.iterdir() if path.is_dir()):
            if study_path.name.startswith("."):
                continue
            names.append(study_path.name)
    return tuple(_unique_preserve_order(names))


def _workspace_protocol_names(
    study_paths: Mapping[str, Path],
    layout_root: Path,
    discovery: Any,
) -> tuple[str, ...]:
    '''Map workspace study folders to bundled feature-policy names.

    Parameters
    ----------
    study_paths : Mapping[str, Path]
        Workspace study folders keyed by policy or study name.
    layout_root : Path
        Resolved OCScore layout root.
    discovery : Any
        Feature policy discovery payload.

    Returns
    -------
    tuple[str, ...]
        Policy names for the baseline (``full_ocscore``) and each ablation study folder.
    '''
    names: list[str] = []
    if _layout_has_baseline_replicas(layout_root) and DEFAULT_REFERENCE_POLICY in discovery.policies:
        names.append(DEFAULT_REFERENCE_POLICY)
    for study_name in sorted(study_paths):
        if study_name in {DEFAULT_REFERENCE_POLICY, "baseline"}:
            continue
        names.append(study_name)
    return tuple(_unique_preserve_order(names))


def _selected_protocol_names(
    discovery: Any,
    *,
    workspace_names: Sequence[str],
    include_catalog_only: bool,
) -> list[str]:
    '''Choose which protocol names to expand for similarity analysis.

    Parameters
    ----------
    discovery : Any
        Feature policy discovery payload.
    workspace_names : Sequence[str]
        Policy names inferred from the served OCScore workspace.
    include_catalog_only : bool
        When true, append bundled catalog policies not present in the workspace.

    Returns
    -------
    list[str]
        Ordered protocol names to resolve.
    '''
    if workspace_names:
        selected = list(workspace_names)
    else:
        selected = list(discovery.available_names)
    if include_catalog_only:
        for name in discovery.available_names:
            if name not in selected:
                selected.append(name)
    elif (
        workspace_names
        and DEFAULT_REFERENCE_POLICY in discovery.policies
        and DEFAULT_REFERENCE_POLICY not in selected
    ):
        selected.append(DEFAULT_REFERENCE_POLICY)
    return _unique_preserve_order(selected)


def _ensure_reference_policy(
    names: Sequence[str],
    discovery: Any,
    *,
    reference_policy: str | None,
) -> list[str]:
    '''Keep the requested reference policy in a filtered protocol name list.

    Parameters
    ----------
    names : Sequence[str]
        Candidate protocol names after workspace filtering.
    discovery : Any
        Feature policy discovery payload with a ``policies`` mapping.
    reference_policy : str or None
        Requested reference policy name.

    Returns
    -------
    list[str]
        Names with the resolved reference policy appended when needed.
    '''
    output = list(names)
    ref_candidate = reference_policy or DEFAULT_REFERENCE_POLICY
    if ref_candidate in discovery.policies and ref_candidate not in output:
        output.append(ref_candidate)
    elif DEFAULT_REFERENCE_POLICY in discovery.policies and DEFAULT_REFERENCE_POLICY not in output:
        output.append(DEFAULT_REFERENCE_POLICY)
    return output


def _prefix_family_id(feature_name: str) -> str | None:
    '''Derive a prefix family token such as ``vina_*`` from one feature column.

    Parameters
    ----------
    feature_name : str
        Expanded candidate feature name.

    Returns
    -------
    str or None
        Prefix family identifier, or ``None`` when the name has no underscore token.
    '''
    token = feature_name.split("_", 1)[0]
    if not token:
        return None
    return f"{token}_*"


def _build_family_definitions(candidate_features: Sequence[str]) -> tuple[_FamilyDefinition, ...]:
    '''Build coarse and prefix feature families for visualization rollups.

    Parameters
    ----------
    candidate_features : Sequence[str]
        Shared candidate columns used by every protocol.

    Returns
    -------
    tuple[_FamilyDefinition, ...]
        Family definitions with member sets drawn from the candidate universe.
    '''
    discovery = _feature_policy_tools()["discover_candidate_model_features"](list(candidate_features))
    blocks = discovery.blocks
    coarse_members: dict[str, set[str]] = {
        "ligand": set(blocks.ligand),
        "receptor": set(blocks.receptor),
        "scoring": set(blocks.scoring),
        "unmatched": set(blocks.unmatched),
    }
    prefix_members: dict[str, set[str]] = {}
    for feature in candidate_features:
        prefix_id = _prefix_family_id(feature)
        if prefix_id is None:
            continue
        prefix_members.setdefault(prefix_id, set()).add(feature)

    definitions: list[_FamilyDefinition] = []
    for family_id in COARSE_FAMILY_IDS:
        members = coarse_members.get(family_id, set())
        if members:
            definitions.append(_FamilyDefinition(family_id=family_id, members=frozenset(members)))
    for family_id in sorted(prefix_members):
        if family_id.endswith("_*") and family_id.split("_", 1)[0] in {"ligand", "receptor"}:
            continue
        members = prefix_members[family_id]
        if members:
            definitions.append(_FamilyDefinition(family_id=family_id, members=frozenset(members)))
    return tuple(definitions)


def _family_states(
    expanded: frozenset[str],
    definitions: Sequence[_FamilyDefinition],
) -> tuple[WorkbenchAblationProtocolFamilyState, ...]:
    '''Summarize which families are present in one expanded protocol.

    Parameters
    ----------
    expanded : frozenset[str]
        Expanded feature set for one policy.
    definitions : Sequence[_FamilyDefinition]
        Family definitions built from the candidate universe.

    Returns
    -------
    tuple[WorkbenchAblationProtocolFamilyState, ...]
        Per-family presence and member counts.
    '''
    states: list[WorkbenchAblationProtocolFamilyState] = []
    for definition in definitions:
        matched = definition.members & expanded
        states.append(
            WorkbenchAblationProtocolFamilyState(
                family_id=definition.family_id,
                present=bool(matched),
                member_count=len(matched),
                total_members=len(definition.members),
            )
        )
    return tuple(states)


def _resolve_protocol_for_name(
    name: str,
    *,
    study_path: Path | None,
    discovery: Any,
    shared_candidates: Sequence[str],
    definitions: Sequence[_FamilyDefinition],
) -> tuple[_ResolvedProtocol | None, InventoryIssue | None]:
    '''Resolve one protocol from study metadata or bundled OCDocker feature policy.

    Executed studies prefer ``final_candidate_features_before_reduction`` from
    one ``feature_policy_metadata.json`` file. Catalog-only policies use
    :func:`OCDocker.OCScore.Utils.FeaturePolicy.apply_feature_policy`.

    Parameters
    ----------
    name : str
        Bundled policy or study name.
    study_path : Path or None
        Study directory when the policy has workspace output.
    discovery : Any
        Feature policy discovery payload.
    shared_candidates : Sequence[str]
        Workspace candidate feature universe for bundled-policy expansion.
    definitions : Sequence[_FamilyDefinition]
        Family definitions for rollup states.

    Returns
    -------
    tuple[_ResolvedProtocol or None, InventoryIssue or None]
        Resolved protocol and optional issue when resolution fails.
    '''
    if study_path is not None:
        loaded = _load_study_feature_policy_metadata(study_path)
        if loaded is not None:
            metadata, metadata_path = loaded
            resolved = _resolved_protocol_from_metadata(name, metadata, metadata_path, definitions)
            if resolved is not None:
                return resolved, None
            study_candidates = _candidate_features_from_metadata(metadata) or list(shared_candidates)
            policy = _policy_from_metadata(metadata, name, discovery)
            if policy is not None and study_candidates:
                try:
                    return _resolve_protocol(policy, study_candidates, definitions), None
                except ValueError as exc:
                    issue = InventoryIssue(path=policy.source_path, message=f"{name}: {exc}")
                    if discovery.policies.get(name) is None:
                        return None, issue

    policy = discovery.policies.get(name)
    if policy is None:
        return None, InventoryIssue(
            path=study_path or Path(name),
            message=f"No feature policy definition found for {name!r}.",
        )
    if not shared_candidates:
        return None, InventoryIssue(
            path=policy.source_path,
            message=f"{name}: no workspace candidate feature list is available for bundled policy expansion.",
        )
    try:
        return _resolve_protocol(policy, shared_candidates, definitions), None
    except ValueError as exc:
        return None, InventoryIssue(path=policy.source_path, message=f"{name}: {exc}")


def _resolve_protocol(
    policy: Any,
    candidate_features: Sequence[str],
    definitions: Sequence[_FamilyDefinition],
) -> _ResolvedProtocol:
    '''Apply one feature policy and capture its expanded set and families.

    Parameters
    ----------
    policy : Any
        Feature policy object from :func:`OCDocker.OCScore.Utils.FeaturePolicy.discover_feature_policies`.
    candidate_features : Sequence[str]
        Shared candidate columns.
    definitions : Sequence[_FamilyDefinition]
        Family definitions for rollup states.

    Returns
    -------
    _ResolvedProtocol
        Resolved protocol metadata and expanded feature set.

    Raises
    ------
    ValueError
        Propagated when policy application fails.
    '''
    application = _feature_policy_tools()["apply_feature_policy"](policy, candidate_features)
    expanded = frozenset(application.final_candidate_features_before_reduction)
    return _ResolvedProtocol(
        policy_name=policy.name,
        policy_description=policy.description,
        policy_source_kind=policy.source_kind,
        policy_source_path=policy.source_path,
        expanded_features=expanded,
        family_states=_family_states(expanded, definitions),
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    '''Compute Jaccard similarity between two expanded feature sets.

    Parameters
    ----------
    left : frozenset[str]
        First expanded feature set.
    right : frozenset[str]
        Second expanded feature set.

    Returns
    -------
    float
        Similarity in ``[0.0, 1.0]``. Empty sets compare as identical.
    '''
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _similarity_matrix(sets: Sequence[frozenset[str]]) -> list[list[float]]:
    '''Build a symmetric pairwise Jaccard similarity matrix.

    Parameters
    ----------
    sets : Sequence[frozenset[str]]
        Expanded feature sets in protocol order.

    Returns
    -------
    list[list[float]]
        Square similarity matrix with ones on the diagonal.
    '''
    size = len(sets)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        matrix[row][row] = 1.0
        for col in range(row + 1, size):
            value = _jaccard(sets[row], sets[col])
            matrix[row][col] = value
            matrix[col][row] = value
    return matrix


def _distance_matrix(similarity: Sequence[Sequence[float]]) -> list[list[float]]:
    '''Convert a similarity matrix to a distance matrix ``1 - similarity``.

    Parameters
    ----------
    similarity : Sequence[Sequence[float]]
        Pairwise similarity matrix.

    Returns
    -------
    list[list[float]]
        Non-negative distance matrix suitable for clustering.
    '''
    return [[max(0.0, 1.0 - similarity[row][col]) for col in range(len(similarity))] for row in range(len(similarity))]


def _cluster_labels(distance: Sequence[Sequence[float]]) -> tuple[list[int], list[int]]:
    '''Cluster protocols and order them for heatmap display.

    Uses SciPy hierarchical clustering when available; otherwise falls back to
    a pure-Python agglomerative merge.

    Parameters
    ----------
    distance : Sequence[Sequence[float]]
        Pairwise distance matrix.

    Returns
    -------
    tuple[list[int], list[int]]
        Cluster label per protocol and leaf order indices for visualization.
    '''
    size = len(distance)
    if size == 0:
        return [], []
    if size == 1:
        return [0], [0]

    try:
        from scipy.cluster.hierarchy import fcluster
        from scipy.cluster.hierarchy import linkage
        from scipy.cluster.hierarchy import leaves_list
        from scipy.spatial.distance import squareform
    except ImportError:
        return _cluster_labels_pure(distance)

    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    order = list(leaves_list(linkage_matrix))
    max_clusters = min(max(2, size // 2), 8)
    labels = list(fcluster(linkage_matrix, t=max_clusters, criterion="maxclust"))
    normalized = [label - 1 for label in labels]
    return normalized, order


def _cluster_labels_pure(distance: Sequence[Sequence[float]]) -> tuple[list[int], list[int]]:
    '''Cluster protocols without SciPy using single-linkage agglomeration.

    Parameters
    ----------
    distance : Sequence[Sequence[float]]
        Pairwise distance matrix.

    Returns
    -------
    tuple[list[int], list[int]]
        Cluster label per protocol and a stable display order.
    '''
    size = len(distance)
    labels = list(range(size))
    order = list(range(size))
    if size <= 2:
        return labels, order

    active = {index for index in range(size)}
    cluster_map = {index: index for index in range(size)}
    next_cluster = size
    while len(active) > 1:
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")
        active_list = sorted(active)
        for left_index, left in enumerate(active_list):
            for right in active_list[left_index + 1 :]:
                dist = distance[left][right]
                if dist < best_distance:
                    best_distance = dist
                    best_pair = (left, right)
        if best_pair is None:
            break
        left, right = best_pair
        merged = next_cluster
        next_cluster += 1
        for index in range(size):
            if cluster_map[index] in {left, right}:
                cluster_map[index] = merged
        active.discard(left)
        active.discard(right)
        active.add(merged)

    normalized_map: dict[int, int] = {}
    normalized_labels: list[int] = []
    next_label = 0
    for index in range(size):
        cluster_id = cluster_map[index]
        if cluster_id not in normalized_map:
            normalized_map[cluster_id] = next_label
            next_label += 1
        normalized_labels.append(normalized_map[cluster_id])

    order = sorted(range(size), key=lambda index: (normalized_labels[index], index))
    return normalized_labels, order


def _reference_diff(
    reference: frozenset[str],
    candidate: frozenset[str],
    definitions: Sequence[_FamilyDefinition],
) -> WorkbenchAblationProtocolReferenceDiff:
    '''Compare one protocol against the reference expanded feature set.

    Parameters
    ----------
    reference : frozenset[str]
        Reference protocol expanded features.
    candidate : frozenset[str]
        Candidate protocol expanded features.
    definitions : Sequence[_FamilyDefinition]
        Family definitions used for family-level add/remove rollups.

    Returns
    -------
    WorkbenchAblationProtocolReferenceDiff
        Added/removed features and families relative to the reference.
    '''
    added_features = sorted(candidate - reference)
    removed_features = sorted(reference - candidate)
    added_families: list[str] = []
    removed_families: list[str] = []
    for definition in definitions:
        ref_present = bool(definition.members & reference)
        cand_present = bool(definition.members & candidate)
        if cand_present and not ref_present:
            added_families.append(definition.family_id)
        elif ref_present and not cand_present:
            removed_families.append(definition.family_id)
    return WorkbenchAblationProtocolReferenceDiff(
        added_features=tuple(added_features),
        removed_features=tuple(removed_features),
        added_families=tuple(added_families),
        removed_families=tuple(removed_families),
        shared_feature_count=len(reference & candidate),
    )


def _metric_summary_mean(summary: Mapping[str, Any], metric_name: str) -> float | None:
    '''Return the aggregate mean for one metric key from an OCScore summary dict.

    Parameters
    ----------
    summary : Mapping[str, Any]
        Study ``metric_summary`` payload.
    metric_name : str
        Scoped or base metric name from the dashboard toolbar.

    Returns
    -------
    float or None
        Aggregate mean when present and numeric.
    '''
    if not summary:
        return None
    entry = summary.get(metric_name)
    if entry is None:
        base = re.sub(r"^(test|validation)_", "", metric_name, count=1)
        if base != metric_name:
            entry = summary.get(base)
    if not isinstance(entry, Mapping):
        return None
    raw = entry.get("mean")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _read_study_aggregate_metrics(study_path: Path) -> dict[str, dict[str, Any]]:
    '''Read aggregate metric means from study-level or first-replica protocol JSON.

    Parameters
    ----------
    study_path : Path
        Baseline or ablation study root.

    Returns
    -------
    dict[str, dict[str, Any]]
        Metric summary compatible with :func:`_metric_summary_mean`.
    '''
    for filename in _STUDY_PROTOCOL_JSON:
        for candidate in _iter_study_file_paths(study_path, filename):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            aggregate = payload.get("aggregate_summary")
            if not isinstance(aggregate, dict):
                continue
            metrics = aggregate.get("metrics")
            if not isinstance(metrics, dict) or not metrics:
                continue
            summary: dict[str, dict[str, Any]] = {}
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, Mapping):
                    summary[str(metric_name)] = dict(metric_data)
                elif isinstance(metric_data, (int, float)):
                    summary[str(metric_name)] = {"mean": float(metric_data)}
            if summary:
                return summary

    for candidate in _iter_study_file_paths(study_path, "summary.json"):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        aggregate = payload.get("aggregate_summary")
        if not isinstance(aggregate, dict):
            continue
        metrics = aggregate.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            continue
        summary = {}
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, Mapping):
                summary[str(metric_name)] = dict(metric_data)
            elif isinstance(metric_data, (int, float)):
                summary[str(metric_name)] = {"mean": float(metric_data)}
        if summary:
            return summary
    return {}


def _layout_study_paths(layout_root: Path) -> tuple[tuple[str, str, Path], ...]:
    '''Return study tuples as ``(study_name, policy_name, path)`` from layout folders.

    Parameters
    ----------
    layout_root : Path
        Resolved OCScore layout root.

    Returns
    -------
    tuple[tuple[str, str, Path], ...]
        Baseline and ablation study roots discovered on disk.
    '''
    studies: list[tuple[str, str, Path]] = []
    if _layout_has_baseline_replicas(layout_root):
        studies.append(("baseline", "baseline", layout_root))
    for container in ablation_container_paths(layout_root):
        if not container.is_dir():
            continue
        for study_path in sorted(path for path in container.iterdir() if path.is_dir()):
            if study_path.name.startswith("."):
                continue
            studies.append((study_path.name, study_path.name, study_path))
    return tuple(studies)


def _fast_workspace_policy_metric_values(
    layout_root: Path,
    *,
    metric_key: str,
) -> tuple[dict[str, float], tuple[InventoryIssue, ...]]:
    '''Load policy metrics from study-level protocol JSON without a full workspace scan.

    Parameters
    ----------
    layout_root : Path
        Resolved OCScore layout root.
    metric_key : str
        Metric name from a parsed comparison objective.

    Returns
    -------
    tuple[dict[str, float], tuple[InventoryIssue, ...]]
        Metric values keyed by study or policy name.
    '''
    values: dict[str, float] = {}
    for study_name, policy_name, study_path in _layout_study_paths(layout_root):
        mean = _metric_summary_mean(_read_study_aggregate_metrics(study_path), metric_key)
        if mean is None:
            continue
        values[study_name] = mean
        if policy_name not in values:
            values[policy_name] = mean
    if "baseline" in values:
        values.setdefault(DEFAULT_REFERENCE_POLICY, values["baseline"])
    return values, ()


def _manifest_policy_metric_values(
    layout_root: Path,
    *,
    metric_key: str,
    max_depth: int,
) -> tuple[dict[str, float], tuple[InventoryIssue, ...]]:
    '''Load policy metrics from shallow Workbench result manifests.

    Parameters
    ----------
    layout_root : Path
        Resolved OCScore layout root.
    metric_key : str
        Metric name from a parsed comparison objective.
    max_depth : int
        Maximum manifest scan depth below the layout root.

    Returns
    -------
    tuple[dict[str, float], tuple[InventoryIssue, ...]]
        Metric values keyed by policy name and any scan issues.
    '''
    from OCDocker.Workbench.Ablation import _flatten_metrics
    from OCDocker.Workbench.Ablation import _policy_from_path
    from OCDocker.Workbench.Ablation import _source_path_from_run_manifest
    from OCDocker.Workbench.Registry import discover_result_manifest_paths
    from OCDocker.Workbench.Registry import read_result_manifest

    values: dict[str, float] = {}
    issues: list[InventoryIssue] = []
    for manifest_path in discover_result_manifest_paths(layout_root, max_depth=max_depth):
        try:
            manifest = read_result_manifest(manifest_path)
        except Exception as exc:
            issues.append(InventoryIssue(path=manifest_path, message=str(exc)))
            continue
        source_path = _source_path_from_run_manifest(manifest_path)
        policy_name, is_ablation = _policy_from_path(source_path, manifest.run_id)
        if not is_ablation:
            continue
        metrics = _flatten_metrics(manifest.metrics)
        raw = metrics.get(metric_key)
        if raw is None:
            continue
        try:
            values[policy_name] = float(raw)
        except (TypeError, ValueError):
            continue
    return values, tuple(issues)


def _policy_metric_values(
    layout_root: Path,
    metric_name: str,
    *,
    max_depth: int,
) -> tuple[dict[str, float], tuple[InventoryIssue, ...]]:
    '''Load one comparison metric per ablation policy without a full workspace scan.

    Parameters
    ----------
    layout_root : Path
        Resolved OCScore layout root used for study-level metric reads.
    metric_name : str
        Comparison metric expression parsed by :func:`OCDocker.Workbench.Ablation.parse_ablation_metric`.
    max_depth : int
        Maximum manifest scan depth when study-level JSON is unavailable.

    Returns
    -------
    tuple[dict[str, float], tuple[InventoryIssue, ...]]
        Metric value keyed by policy name and any scan or parse issues.
    '''
    try:
        objective = parse_ablation_metric(metric_name)
    except ValueError as exc:
        return {}, (InventoryIssue(path=layout_root, message=str(exc)),)

    values, issues = _fast_workspace_policy_metric_values(
        layout_root,
        metric_key=objective.metric_name,
    )
    manifest_values, manifest_issues = _manifest_policy_metric_values(
        layout_root,
        metric_key=objective.metric_name,
        max_depth=min(max_depth, 2),
    )
    for name, value in manifest_values.items():
        values.setdefault(name, value)
    return values, (*issues, *manifest_issues)


def _cluster_summaries(
    protocol_names: Sequence[str],
    cluster_labels: Sequence[int],
    metric_values: Mapping[str, float | None],
) -> tuple[WorkbenchAblationProtocolClusterSummary, ...]:
    '''Aggregate optional outcome metrics per feature-similarity cluster.

    Parameters
    ----------
    protocol_names : Sequence[str]
        Protocol names aligned with cluster labels.
    cluster_labels : Sequence[int]
        Cluster id assigned to each protocol.
    metric_values : Mapping[str, float | None]
        Optional metric overlay keyed by policy name.

    Returns
    -------
    tuple[WorkbenchAblationProtocolClusterSummary, ...]
        One summary row per cluster, sorted by cluster id.
    '''
    clusters: dict[int, list[str]] = {}
    for name, label in zip(protocol_names, cluster_labels, strict=True):
        clusters.setdefault(label, []).append(name)

    summaries: list[WorkbenchAblationProtocolClusterSummary] = []
    for cluster_id in sorted(clusters):
        names = tuple(sorted(clusters[cluster_id]))
        present_values = [value for name in names if (value := metric_values.get(name)) is not None]
        mean_metric = sum(present_values) / len(present_values) if present_values else None
        summaries.append(
            WorkbenchAblationProtocolClusterSummary(
                cluster_id=cluster_id,
                policy_names=names,
                mean_metric=mean_metric,
                metric_count=len(present_values),
                missing_metric_count=len(names) - len(present_values),
            )
        )
    return tuple(summaries)


## Public ##


def build_ablation_protocol_similarity_analysis(
    root: str | Path,
    *,
    reference_policy: str | None = None,
    metric: str | None = None,
    include_catalog_only: bool = False,
    max_depth: int = 6,
) -> WorkbenchAblationProtocolSimilarity:
    '''Build protocol similarity analysis for one Workbench root.

    Each selected policy is expanded with the same candidate feature universe,
    compared with Jaccard similarity, clustered for visualization, and annotated
    with optional outcome metrics and reference diffs.

    Parameters
    ----------
    root : str or pathlib.Path
        Workbench root to scan.
    reference_policy : str or None
        Reference policy name for diffs. Defaults to ``full_ocscore`` when present.
    metric : str or None
        Optional comparison metric used for cluster mean overlays.
    include_catalog_only : bool
        When ``True``, always include the full bundled and workspace policy catalog.
        When ``False``, restrict to policies with executed workspace folders or
        ablation result manifests. If none are detected, the full catalog is still
        returned so feature-level similarity remains usable.
    max_depth : int
        Maximum scan depth for result manifests when resolving metric overlays.

    Returns
    -------
    WorkbenchAblationProtocolSimilarity
        Similarity analysis payload for API and dashboard consumers.
    '''

    root_path = Path(root)
    layout_root = resolve_ocscore_layout_root(root_path)
    study_paths = _study_paths_by_policy(layout_root)
    candidates, candidate_source = _resolve_workspace_candidate_features(root_path, layout_root, study_paths)
    preview_available = bool(candidates) or any(
        _load_study_feature_policy_metadata(study_path) is not None for study_path in study_paths.values()
    ) or bool(study_paths)

    policy_dirs = _workspace_policy_dirs(layout_root)
    discovery = _feature_policy_tools()["discover_feature_policies"](policy_dirs=policy_dirs or None)
    workspace_names = _workspace_protocol_names(study_paths, layout_root, discovery)
    selected_names = _selected_protocol_names(
        discovery,
        workspace_names=workspace_names,
        include_catalog_only=include_catalog_only,
    )

    reference_name = reference_policy or DEFAULT_REFERENCE_POLICY
    if reference_name not in discovery.policies:
        if DEFAULT_REFERENCE_POLICY in discovery.policies:
            reference_name = DEFAULT_REFERENCE_POLICY
        elif selected_names:
            reference_name = selected_names[0]
        else:
            reference_name = DEFAULT_REFERENCE_POLICY

    issues: list[InventoryIssue] = []
    metric_values: dict[str, float | None] = {name: None for name in selected_names}
    if metric:
        resolved_metrics, metric_issues = _policy_metric_values(
            layout_root,
            metric,
            max_depth=max_depth,
        )
        issues.extend(metric_issues)
        for name, value in resolved_metrics.items():
            if name in metric_values:
                metric_values[name] = value

    if not preview_available:
        return WorkbenchAblationProtocolSimilarity(
            root=root_path,
            layout_root=layout_root,
            candidate_source=candidate_source,
            preview_available=False,
            reference_policy=reference_name,
            metric=metric or "",
            include_catalog_only=include_catalog_only,
            protocol_count=0,
            protocols=(),
            protocol_order=(),
            similarity_matrix=(),
            cluster_labels=(),
            cluster_summaries=(),
            reference_diffs=(),
            issue_count=len(issues),
            issues=tuple(issues),
            message=(
                "No candidate feature list was found in this workspace. "
                "Load raw_prepare tables or run training to compare expanded protocols."
            ),
        )

    definition_candidates = list(candidates)
    if not definition_candidates:
        for study_path in study_paths.values():
            loaded = _load_study_feature_policy_metadata(study_path)
            if loaded is None:
                continue
            expanded = _expanded_features_from_metadata(loaded[0])
            if expanded:
                definition_candidates = sorted(expanded)
                break
    definitions = _build_family_definitions(definition_candidates)
    resolved_by_name: dict[str, _ResolvedProtocol] = {}
    for name in selected_names:
        item, issue = _resolve_protocol_for_name(
            name,
            study_path=study_paths.get(name),
            discovery=discovery,
            shared_candidates=candidates,
            definitions=definitions,
        )
        if item is not None:
            resolved_by_name[item.policy_name] = item
        elif issue is not None:
            issues.append(issue)

    resolved = [resolved_by_name[name] for name in selected_names if name in resolved_by_name]

    if not resolved:
        return WorkbenchAblationProtocolSimilarity(
            root=root_path,
            layout_root=layout_root,
            candidate_source=candidate_source,
            preview_available=True,
            reference_policy=reference_name,
            metric=metric or "",
            include_catalog_only=include_catalog_only,
            protocol_count=0,
            protocols=(),
            protocol_order=(),
            similarity_matrix=(),
            cluster_labels=(),
            cluster_summaries=(),
            reference_diffs=(),
            issue_count=len(issues),
            issues=tuple(issues),
            message="No protocols could be resolved against the candidate feature list.",
        )

    reference_resolved = None
    for item in resolved:
        if item.policy_name == reference_name:
            reference_resolved = item
            break
    if reference_resolved is None and (reference_name in discovery.policies or reference_name in study_paths):
        ref_item, ref_issue = _resolve_protocol_for_name(
            reference_name,
            study_path=study_paths.get(reference_name),
            discovery=discovery,
            shared_candidates=candidates,
            definitions=definitions,
        )
        if ref_item is not None:
            reference_resolved = ref_item
        elif ref_issue is not None:
            issues.append(ref_issue)

    reference_set = reference_resolved.expanded_features if reference_resolved is not None else frozenset()

    protocol_entries: list[WorkbenchAblationProtocolSimilarityEntry] = []
    expanded_sets: list[frozenset[str]] = []
    for item in resolved:
        item_study_path = study_paths.get(item.policy_name)
        protocol_entries.append(
            WorkbenchAblationProtocolSimilarityEntry(
                policy_name=item.policy_name,
                description=item.policy_description,
                source_kind=item.policy_source_kind,
                source_path=item.policy_source_path,
                expanded_feature_count=len(item.expanded_features),
                run_id=_completed_run_id(item_study_path, layout_root=layout_root, policy_name=item.policy_name),
                study_present=item_study_path is not None and item_study_path.is_dir(),
                metric_value=metric_values.get(item.policy_name),
                families=item.family_states,
            )
        )
        expanded_sets.append(item.expanded_features)

    similarity = _similarity_matrix(expanded_sets)
    distance = _distance_matrix(similarity)
    cluster_labels, order = _cluster_labels(distance)
    ordered_names = [protocol_entries[index].policy_name for index in order]
    ordered_similarity = [[similarity[row][col] for col in order] for row in order]
    ordered_clusters = [cluster_labels[index] for index in order]

    reference_diffs: list[WorkbenchAblationProtocolReferenceDiff] = []
    for item in resolved:
        diff = _reference_diff(reference_set, item.expanded_features, definitions)
        reference_diffs.append(
            WorkbenchAblationProtocolReferenceDiff(
                policy_name=item.policy_name,
                added_features=(),
                removed_features=(),
                added_families=diff.added_families,
                removed_families=diff.removed_families,
                shared_feature_count=diff.shared_feature_count,
            )
        )

    cluster_summaries = _cluster_summaries(
        [entry.policy_name for entry in protocol_entries],
        cluster_labels,
        metric_values,
    )

    resolved_count = len(protocol_entries)
    selected_count = len(selected_names)
    workspace_study_count = sum(
        1 for name in study_paths if name not in {DEFAULT_REFERENCE_POLICY, "baseline"}
    )
    message = ""
    if resolved_count < selected_count:
        message = (
            f"Expanded {resolved_count} of {selected_count} protocols "
            f"({workspace_study_count} ablation folders); "
            f"{selected_count - resolved_count} skipped (see issues)."
        )

    return WorkbenchAblationProtocolSimilarity(
        root=root_path,
        layout_root=layout_root,
        candidate_source=candidate_source,
        preview_available=True,
        reference_policy=reference_name,
        metric=metric or "",
        include_catalog_only=include_catalog_only,
        protocol_count=len(protocol_entries),
        protocols=tuple(protocol_entries),
        protocol_order=tuple(ordered_names),
        similarity_matrix=tuple(tuple(row) for row in ordered_similarity),
        cluster_labels=tuple(ordered_clusters),
        cluster_summaries=cluster_summaries,
        reference_diffs=tuple(reference_diffs),
        issue_count=len(issues),
        issues=tuple(issues),
        message=message,
    )


__all__ = [
    "build_ablation_protocol_similarity_analysis",
]
