#!/usr/bin/env python3

# Description
###############################################################################
'''
Feature-ablation policy discovery and application for OCScore training.

Policies constrain the candidate descriptor pool before train-only feature
cleaning and reduction are fitted.
'''

from __future__ import annotations

# Imports
###############################################################################
import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import yaml

import OCDocker.OCScore.Utils.FeatureReduction as ocfr
import OCDocker.Toolbox.Logging as oclogging

from OCDocker.OCScore.Utils.ContentHash import hash_feature_list
from OCDocker.OCScore.Utils.ContentHash import hash_file

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

LOGGER = oclogging.get_logger("ocscore.utils.feature_policy")

BUNDLED_FEATURE_POLICY_DIR = Path(__file__).resolve().parent.parent / "Protocols" / "Ablations"
FULL_OCSCORE_POLICY_NAME = "full_ocscore"
FEATURE_POLICY_METADATA_JSON = "feature_policy_metadata.json"
FEATURE_POLICY_SUMMARY_JSON = "feature_policy_ablation_summary.json"
FEATURE_POLICY_SUMMARY_CSV = "feature_policy_ablation_summary.csv"
VALID_SOURCE_KINDS = ("bundled", "user_dir", "explicit_yml")


# Classes
###############################################################################


@dataclass(frozen=True)
class FeaturePolicy:
    """Loaded feature-ablation policy."""

    name: str
    description: str = ""
    include_features: tuple[str, ...] = ()
    include_patterns: tuple[str, ...] = ()
    exclude_features: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    allow_missing_exclude_features: bool = True
    allow_empty_policy: bool = False
    source_path: Path = Path()
    source_kind: str = "bundled"
    source_hash: str = ""

    def request_payload(self) -> dict[str, Any]:
        '''Serialize policy fields for provenance and CLI replay.

        Returns
        -------
        dict[str, Any]
            JSON-serializable policy request payload.
        '''

        return {
            "name": self.name,
            "description": self.description,
            "include_features": list(self.include_features),
            "include_patterns": list(self.include_patterns),
            "exclude_features": list(self.exclude_features),
            "exclude_patterns": list(self.exclude_patterns),
            "allow_missing_exclude_features": self.allow_missing_exclude_features,
            "allow_empty_policy": self.allow_empty_policy,
        }


@dataclass(frozen=True)
class FeaturePolicyDiscovery:
    """Resolved feature-policy lookup pool."""

    policies: dict[str, FeaturePolicy]
    lookup_dirs: tuple[Path, ...]
    explicit_ymls: tuple[Path, ...] = ()

    @property
    def available_names(self) -> tuple[str, ...]:
        '''Sorted policy names discovered in the lookup pool.

        Returns
        -------
        tuple[str, ...]
            Policy names available for ``--feature-policy`` resolution.
        '''

        return tuple(self.policies.keys())


@dataclass(frozen=True)
class CandidateFeatureDiscovery:
    """Candidate model features before policy filtering."""

    candidate_features: list[str]
    metadata_columns: list[str]
    target_columns: list[str]
    unmatched_columns: list[str]
    duplicate_assignments: dict[str, list[str]]
    blocks: ocfr.DescriptorBlocks


@dataclass(frozen=True)
class FeaturePolicyApplication:
    """Result of applying one policy to candidate model features."""

    policy: FeaturePolicy
    lookup_dirs: tuple[Path, ...]
    candidate_features_before_policy: list[str]
    included_features_found: list[str]
    excluded_features_found: list[str]
    missing_include_features: list[str]
    missing_exclude_features: list[str]
    patterns_with_no_matches: list[str]
    final_candidate_features_before_reduction: list[str]

    def to_metadata(self) -> dict[str, Any]:
        '''Build provenance metadata for one policy application result.

        Returns
        -------
        dict[str, Any]
            Feature-policy audit fields written beside training artifacts.
        '''

        final_candidates = list(self.final_candidate_features_before_reduction)
        candidates = list(self.candidate_features_before_policy)
        return {
            "feature_policy_name": self.policy.name,
            "feature_policy_description": self.policy.description,
            "feature_policy_source_path": str(self.policy.source_path),
            "feature_policy_source_kind": self.policy.source_kind,
            "feature_policy_hash": self.policy.source_hash,
            "feature_policy_lookup_dirs": [str(path) for path in self.lookup_dirs],
            "candidate_feature_count_before_policy": len(candidates),
            "candidate_features_before_policy_hash": hash_feature_list(candidates),
            "included_features_requested": list(self.policy.include_features),
            "included_patterns_requested": list(self.policy.include_patterns),
            "excluded_features_requested": list(self.policy.exclude_features),
            "excluded_patterns_requested": list(self.policy.exclude_patterns),
            "included_features_found": list(self.included_features_found),
            "excluded_features_found": list(self.excluded_features_found),
            "missing_include_features": list(self.missing_include_features),
            "missing_exclude_features": list(self.missing_exclude_features),
            "patterns_with_no_matches": list(self.patterns_with_no_matches),
            "final_candidate_features_before_reduction": final_candidates,
            "final_candidate_features_before_reduction_hash": hash_feature_list(final_candidates),
        }


# Functions
###############################################################################
## Private ##


def _as_str_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise ValueError(f"Feature policy field {field_name!r} must be a list of strings, not a string.")
    if not isinstance(value, Sequence):
        raise ValueError(f"Feature policy field {field_name!r} must be a list of strings.")
    return tuple(str(item) for item in value)


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _feature_policy_error_list(names: Sequence[str]) -> str:
    return "\n".join(f"- {name}" for name in names)


def _format_policy_conflicts(conflicts: Mapping[str, Sequence[FeaturePolicy]]) -> str:
    blocks: list[str] = []
    for name, policies in conflicts.items():
        paths = "\n".join(f"- {policy.source_path}" for policy in policies)
        blocks.append(
            f'Feature policy name conflict: "{name}"\n\n'
            f"Found in:\n{paths}\n\n"
            "Please rename one policy or remove one directory from the lookup pool."
        )
    return "\n\n".join(blocks)


def _iter_policy_files(policy_dir: Path) -> list[Path]:
    if not policy_dir.exists():
        raise FileNotFoundError(f"Feature policy directory does not exist: {policy_dir}")
    if not policy_dir.is_dir():
        raise ValueError(f"Feature policy lookup path is not a directory: {policy_dir}")
    return sorted(path for path in policy_dir.glob("*.yml") if path.is_file())


def _match_patterns(candidate_features: Sequence[str], patterns: Sequence[str]) -> tuple[list[str], list[str]]:
    matched: list[str] = []
    no_match: list[str] = []
    for pattern in patterns:
        pattern_matches = [
            feature
            for feature in candidate_features
            if fnmatch.fnmatchcase(feature, pattern)
        ]
        if pattern_matches:
            matched.extend(pattern_matches)
        else:
            no_match.append(pattern)
    return _unique_preserve_order(matched), no_match


## Public ##


def load_feature_policy(path: str | Path, *, source_kind: str = "bundled") -> FeaturePolicy:
    """Load one feature policy from a ``.yml`` file."""

    if source_kind not in VALID_SOURCE_KINDS:
        raise ValueError(f"Unknown feature policy source kind: {source_kind!r}.")
    policy_path = Path(path).expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"Feature policy file not found: {policy_path}")
    if policy_path.suffix != ".yml":
        raise ValueError(f"Feature policy files must use .yml extension: {policy_path}")

    loaded = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Feature policy {policy_path} must contain a YAML mapping.")
    name = str(loaded.get("name", "")).strip()
    if not name:
        raise ValueError(f"Feature policy {policy_path} is missing required field 'name'.")

    return FeaturePolicy(
        name=name,
        description=str(loaded.get("description", "") or "").strip(),
        include_features=_as_str_tuple(loaded.get("include_features"), "include_features"),
        include_patterns=_as_str_tuple(loaded.get("include_patterns"), "include_patterns"),
        exclude_features=_as_str_tuple(loaded.get("exclude_features"), "exclude_features"),
        exclude_patterns=_as_str_tuple(loaded.get("exclude_patterns"), "exclude_patterns"),
        allow_missing_exclude_features=bool(loaded.get("allow_missing_exclude_features", True)),
        allow_empty_policy=bool(loaded.get("allow_empty_policy", False)),
        source_path=policy_path,
        source_kind=source_kind,
        source_hash=hash_file(policy_path),
    )


def discover_feature_policies(
        *,
        policy_dirs: Optional[Sequence[str | Path]] = None,
        explicit_ymls: Optional[Sequence[str | Path]] = None,
    ) -> FeaturePolicyDiscovery:
    """Discover bundled, user-directory, and explicit feature policies."""

    lookup_dirs = [BUNDLED_FEATURE_POLICY_DIR]
    lookup_dirs.extend(Path(path).expanduser().resolve() for path in (policy_dirs or ()))
    explicit_paths = tuple(Path(path).expanduser().resolve() for path in (explicit_ymls or ()))

    by_name: dict[str, list[FeaturePolicy]] = {}
    for index, directory in enumerate(lookup_dirs):
        source_kind = "bundled" if index == 0 else "user_dir"
        for policy_file in _iter_policy_files(directory):
            policy = load_feature_policy(policy_file, source_kind=source_kind)
            by_name.setdefault(policy.name, []).append(policy)
    for explicit_path in explicit_paths:
        policy = load_feature_policy(explicit_path, source_kind="explicit_yml")
        by_name.setdefault(policy.name, []).append(policy)

    conflicts = {
        name: policies
        for name, policies in by_name.items()
        if len(policies) > 1
    }
    if conflicts:
        raise ValueError(_format_policy_conflicts(conflicts))

    policies = {name: policies[0] for name, policies in by_name.items()}
    return FeaturePolicyDiscovery(
        policies=policies,
        lookup_dirs=tuple(lookup_dirs),
        explicit_ymls=explicit_paths,
    )


def resolve_requested_feature_policies(
        *,
        requested_names: Optional[Sequence[str]] = None,
        run_all: bool = False,
        policy_dirs: Optional[Sequence[str | Path]] = None,
        explicit_ymls: Optional[Sequence[str | Path]] = None,
    ) -> tuple[list[FeaturePolicy], FeaturePolicyDiscovery]:
    """Resolve CLI feature-policy requests to loaded policies."""

    discovery = discover_feature_policies(policy_dirs=policy_dirs, explicit_ymls=explicit_ymls)
    requested = [str(name).strip() for name in (requested_names or ()) if str(name).strip()]
    explicit_names = [
        load_feature_policy(path, source_kind="explicit_yml").name
        for path in (explicit_ymls or ())
    ]

    if run_all:
        if requested:
            raise ValueError("Use either --run-all-feature-policies or --feature-policy, not both.")
        return [discovery.policies[name] for name in discovery.available_names], discovery

    if not requested:
        requested = explicit_names or [FULL_OCSCORE_POLICY_NAME]

    duplicate_requests = sorted(name for name in set(requested) if requested.count(name) > 1)
    if duplicate_requests:
        raise ValueError(f"Duplicate requested feature policy name(s): {duplicate_requests}")

    unknown = [name for name in requested if name not in discovery.policies]
    if unknown:
        available = _feature_policy_error_list(discovery.available_names)
        raise ValueError(
            f'Unknown feature policy: "{unknown[0]}"\n\n'
            f"Available policies:\n{available}"
        )
    return [discovery.policies[name] for name in requested], discovery


def discover_candidate_model_features(
        columns: Sequence[str],
        *,
        config: Optional[ocfr.FeatureReductionConfig] = None,
        non_feature_columns: Optional[Sequence[str]] = None,
    ) -> CandidateFeatureDiscovery:
    """Discover the current production candidate descriptor pool."""

    cfg = config or ocfr.default_ocscore_feature_reduction_config()
    blocks = ocfr.split_descriptor_blocks(
        columns=columns,
        metadata_columns=cfg.block_detection.metadata_columns,
        target_columns=cfg.block_detection.target_columns,
        receptor_patterns=cfg.block_detection.receptor_patterns,
        ligand_patterns=cfg.block_detection.ligand_patterns,
        scoring_patterns=cfg.block_detection.scoring_patterns,
        use_ligand_class_descriptors=cfg.block_detection.use_ligand_class_descriptors,
        use_receptor_class_descriptors=cfg.block_detection.use_receptor_class_descriptors,
        use_scoring_model_descriptors=cfg.block_detection.use_scoring_model_descriptors,
    )
    reserved = set(non_feature_columns or ())
    candidate_features = [
        feature
        for feature in blocks.all_descriptor_columns
        if feature not in reserved
    ]
    return CandidateFeatureDiscovery(
        candidate_features=_unique_preserve_order(candidate_features),
        metadata_columns=list(blocks.metadata),
        target_columns=list(blocks.target),
        unmatched_columns=list(blocks.unmatched),
        duplicate_assignments={key: list(value) for key, value in blocks.duplicate_assignments.items()},
        blocks=blocks,
    )


def apply_feature_policy(
        policy: FeaturePolicy,
        candidate_features: Sequence[str],
        *,
        lookup_dirs: Sequence[str | Path] = (),
    ) -> FeaturePolicyApplication:
    """Apply one feature policy to ordered candidate model features."""

    candidates = _unique_preserve_order(candidate_features)
    candidate_set = set(candidates)

    include_requested = _unique_preserve_order(policy.include_features)
    missing_include = [feature for feature in include_requested if feature not in candidate_set]
    if missing_include:
        raise ValueError(
            f"Feature policy {policy.name!r} requested missing include_features: {missing_include}"
        )

    patterns_with_no_matches: list[str] = []
    include_match_set = set(include_requested)
    if policy.include_patterns:
        pattern_matches, no_match = _match_patterns(candidates, policy.include_patterns)
        include_match_set.update(pattern_matches)
        patterns_with_no_matches.extend(no_match)
    if not include_requested and not policy.include_patterns:
        if policy.name != FULL_OCSCORE_POLICY_NAME:
            raise ValueError(
                f"Feature policy {policy.name!r} must define include_features or include_patterns."
            )
        included = list(candidates)
    else:
        included = [feature for feature in candidates if feature in include_match_set]

    missing_exclude = [
        feature
        for feature in _unique_preserve_order(policy.exclude_features)
        if feature not in candidate_set
    ]
    if missing_exclude:
        message = (
            f"Feature policy {policy.name!r} requested exclude_features not present in candidates: "
            f"{missing_exclude}"
        )
        if policy.allow_missing_exclude_features:
            LOGGER.warning(message)
        else:
            raise ValueError(message)

    excluded: list[str] = []
    exclude_existing = set(
        feature
        for feature in _unique_preserve_order(policy.exclude_features)
        if feature in candidate_set
    )
    if exclude_existing:
        excluded.extend(feature for feature in included if feature in exclude_existing)
    if policy.exclude_patterns:
        pattern_matches, no_match = _match_patterns(candidates, policy.exclude_patterns)
        excluded.extend(feature for feature in included if feature in set(pattern_matches))
        patterns_with_no_matches.extend(no_match)

    for pattern in patterns_with_no_matches:
        LOGGER.warning(
            "Feature policy %r pattern matched no candidate features: %s",
            policy.name,
            pattern,
        )

    excluded = _unique_preserve_order(excluded)
    excluded_set = set(excluded)
    final_candidates = [feature for feature in included if feature not in excluded_set]
    if not final_candidates and not policy.allow_empty_policy:
        raise ValueError(f"Feature policy {policy.name!r} produced an empty candidate feature set.")

    return FeaturePolicyApplication(
        policy=policy,
        lookup_dirs=tuple(Path(path).expanduser().resolve() for path in lookup_dirs),
        candidate_features_before_policy=list(candidates),
        included_features_found=list(included),
        excluded_features_found=excluded,
        missing_include_features=missing_include,
        missing_exclude_features=missing_exclude,
        patterns_with_no_matches=_unique_preserve_order(patterns_with_no_matches),
        final_candidate_features_before_reduction=final_candidates,
    )


def write_feature_policy_metadata(output_dir: str | Path, metadata: Mapping[str, Any]) -> Path:
    """Write feature-policy provenance metadata."""

    path = Path(output_dir) / FEATURE_POLICY_METADATA_JSON
    path.write_text(json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "BUNDLED_FEATURE_POLICY_DIR",
    "FEATURE_POLICY_METADATA_JSON",
    "FEATURE_POLICY_SUMMARY_CSV",
    "FEATURE_POLICY_SUMMARY_JSON",
    "FULL_OCSCORE_POLICY_NAME",
    "CandidateFeatureDiscovery",
    "FeaturePolicy",
    "FeaturePolicyApplication",
    "FeaturePolicyDiscovery",
    "apply_feature_policy",
    "discover_candidate_model_features",
    "discover_feature_policies",
    "load_feature_policy",
    "resolve_requested_feature_policies",
    "write_feature_policy_metadata",
]
