#!/usr/bin/env python3

# Description
###############################################################################
'''Structured YAML protocol files for staged OCScore training.

Protocols define replicas, trial budgets, split/scaling policy, and reporting
artifacts. The train CLI loads a protocol file and applies it without preset
flags or CLI hyperparameter overrides.
'''

# Imports
###############################################################################
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from OCDocker.OCScore.Utils.PDBbindSplit import PDBbindSplitConfig

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

BUNDLED_PROTOCOL_DIR = Path(__file__).resolve().parent.parent / "Protocols"
CalibrationReportMode = Literal["ranking_only", "calibration_validated"]
DUDEzScalingStrategy = Literal["pdbbind_scaler", "dudez_train_scaler", "none_prestandardized"]
PDBbindSearchPhase = Literal["full", "encoder_regression"]
AblationVariantName = Literal[
    "ligand_only",
    "sf_only",
    "ligand_sf",
    "receptor_sf",
]
DEFAULT_ABLATION_VARIANTS: tuple[AblationVariantName, ...] = (
    "ligand_only",
    "sf_only",
    "ligand_sf",
    "receptor_sf",
)
ABLATION_VARIANT_ALIASES: dict[str, AblationVariantName] = {
    "ligand": "ligand_only",
    "ligand-only": "ligand_only",
    "ligand_only": "ligand_only",
    "sf": "sf_only",
    "sf-only": "sf_only",
    "sf_only": "sf_only",
    "scoring": "sf_only",
    "scoring-only": "sf_only",
    "scoring_only": "sf_only",
    "ligand-sf": "ligand_sf",
    "ligand_sf": "ligand_sf",
    "ligand-scoring": "ligand_sf",
    "ligand_scoring": "ligand_sf",
    "receptor-sf": "receptor_sf",
    "receptor_sf": "receptor_sf",
    "receptor-scoring": "receptor_sf",
    "receptor_scoring": "receptor_sf",
}


@dataclass(frozen=True)
class PDBbindProtocolSection:
    """PDBbind regression stage settings from a staged train protocol."""

    target_column: str
    trials: int
    epochs: int
    n_jobs: int
    search_phase: PDBbindSearchPhase
    enable_pruning: bool
    split_strategy: str
    split_train_size: float
    split_validation_size: float
    split_test_size: float


@dataclass(frozen=True)
class DUDEzProtocolSection:
    """DUDEz ranking stage settings from a staged train protocol."""

    kind_column: str
    positive_kind: str
    negative_kind: str
    trials: int
    epochs: int
    n_jobs: int
    primary_metric: str
    bedroc_alpha: float
    scaling_strategy: DUDEzScalingStrategy
    ignore_unknown_kind: bool


@dataclass(frozen=True)
class RuntimeProtocolSection:
    """Execution controls shared across replicas and stages."""

    use_gpu: bool
    pdbbind_only: bool
    replica_jobs: int
    resume_completed: bool


@dataclass(frozen=True)
class ReportingProtocolSection:
    """Optional post-training reports and audits."""

    generate_final_report: bool
    run_leakage_audit: bool
    run_baselines: bool
    calibration_report_mode: CalibrationReportMode


@dataclass(frozen=True)
class AblationProtocolSection:
    """Feature-policy ablation variants to run after baseline replicas."""

    enabled: bool
    variants: tuple[AblationVariantName, ...]


@dataclass(frozen=True)
class ProductionClaimRequirements:
    """Minimum replica and trial budgets enforced for production claims."""

    enforce: bool
    min_replicas: int
    min_pdbbind_trials: int
    min_dudez_trials: int


@dataclass(frozen=True)
class StagedTrainProtocol:
    """Structured staged-training protocol loaded from YAML."""

    name: str
    description: str
    replicas: int
    seed: int
    pdbbind: PDBbindProtocolSection
    dudez: DUDEzProtocolSection
    runtime: RuntimeProtocolSection
    reporting: ReportingProtocolSection
    ablation: AblationProtocolSection
    source_path: Path
    production_claim: Optional[ProductionClaimRequirements] = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def pdbbind_split_config(self) -> PDBbindSplitConfig:
        '''Build the PDBbind split configuration implied by this protocol.

        Returns
        -------
        PDBbindSplitConfig
            Split strategy and sizes for PDBbind regression rows.
        '''

        return PDBbindSplitConfig(
            strategy=self.pdbbind.split_strategy,
            train_size=self.pdbbind.split_train_size,
            validation_size=self.pdbbind.split_validation_size,
            test_size=self.pdbbind.split_test_size,
            random_seed=self.seed,
        )

    def budget_dict(self) -> dict[str, Any]:
        '''Serialize replica and trial budgets for provenance metadata.

        Returns
        -------
        dict[str, Any]
            Budget fields written into training provenance bundles.
        '''

        return {
            "protocol": self.name,
            "protocol_path": str(self.source_path),
            "n_replicas": self.replicas,
            "pdbbind_trials": self.pdbbind.trials,
            "dudez_trials": self.dudez.trials,
            "dudez_bedroc_alpha": self.dudez.bedroc_alpha,
            "pdbbind_epochs": self.pdbbind.epochs,
            "dudez_epochs": self.dudez.epochs,
            "pdbbind_n_jobs": self.pdbbind.n_jobs,
            "dudez_n_jobs": self.dudez.n_jobs,
            "replica_jobs": self.runtime.replica_jobs,
            "resume_completed": self.runtime.resume_completed,
            "ablation_enabled": self.ablation.enabled,
            "ablation_variants": list(self.ablation.variants),
        }

    def validate_production_claim_budget(self) -> None:
        '''Raise when configured budgets fall below production-claim thresholds.

        Raises
        ------
        ValueError
            If ``production_claim.enforce`` is true and replicas or trials are too low.
        '''

        if self.production_claim is None or not self.production_claim.enforce:
            return
        issues: list[str] = []
        if self.replicas < self.production_claim.min_replicas:
            issues.append(f"replicas={self.replicas} < {self.production_claim.min_replicas}")
        if self.pdbbind.trials < self.production_claim.min_pdbbind_trials:
            issues.append(
                f"pdbbind.trials={self.pdbbind.trials} < {self.production_claim.min_pdbbind_trials}"
            )
        if self.dudez.trials < self.production_claim.min_dudez_trials:
            issues.append(
                f"dudez.trials={self.dudez.trials} < {self.production_claim.min_dudez_trials}"
            )
        if issues:
            raise ValueError(
                "Protocol does not meet production-claim budget requirements: "
                + "; ".join(issues)
            )


def bundled_protocol_names() -> tuple[str, ...]:
    '''List bundled protocol stems shipped under ``OCScore/Protocols/``.

    Returns
    -------
    tuple[str, ...]
        Sorted protocol names without file extensions.
    '''

    names: list[str] = []
    for path in sorted(BUNDLED_PROTOCOL_DIR.glob("*.yml")):
        names.append(path.stem)
    for path in sorted(BUNDLED_PROTOCOL_DIR.glob("*.yaml")):
        if path.stem not in names:
            names.append(path.stem)
    return tuple(names)


def resolve_protocol_path(spec: str) -> Path:
    '''Resolve a user path or bundled protocol name to an on-disk YAML file.

    Parameters
    ----------
    spec : str
        Filesystem path or bundled protocol stem (for example ``production``).

    Returns
    -------
    pathlib.Path
        Absolute path to the resolved protocol YAML file.

    Raises
    ------
    ValueError
        If neither a file nor a bundled protocol matches ``spec``.
    '''

    candidate = Path(spec).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    normalized = str(spec).strip()
    for suffix in (".yml", ".yaml", ""):
        bundled = BUNDLED_PROTOCOL_DIR / f"{normalized}{suffix}"
        if bundled.is_file():
            return bundled.resolve()

    available = ", ".join(bundled_protocol_names()) or "(none bundled)"
    raise ValueError(
        f"Protocol file not found for {spec!r}. "
        f"Provide a path to a YAML protocol file or a bundled name: {available}."
    )


def normalize_ablation_variant_name(value: str) -> AblationVariantName:
    '''Normalize CLI or YAML ablation variant aliases to canonical names.

    Parameters
    ----------
    value : str
        Raw variant label from protocol YAML or CLI flags.

    Returns
    -------
    AblationVariantName
        Canonical ablation variant identifier.

    Raises
    ------
    ValueError
        If ``value`` does not match a known alias.
    '''

    normalized = str(value).strip().lower().replace(" ", "_")
    normalized = normalized.replace("+", "_").replace("/", "_")
    if normalized in ABLATION_VARIANT_ALIASES:
        return ABLATION_VARIANT_ALIASES[normalized]
    valid = ", ".join(DEFAULT_ABLATION_VARIANTS)
    raise ValueError(f"Unknown ablation variant {value!r}. Expected one of: {valid}.")


def _load_ablation_section(raw: dict[str, Any]) -> AblationProtocolSection:
    loaded = raw.get("ablation", {})
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError("Protocol field 'ablation' must be a mapping when provided.")
    variant_values = loaded.get("variants", DEFAULT_ABLATION_VARIANTS)
    if variant_values is None:
        variant_values = DEFAULT_ABLATION_VARIANTS
    if isinstance(variant_values, str):
        variant_values = [variant_values]
    variants = tuple(normalize_ablation_variant_name(value) for value in variant_values)
    if not variants:
        raise ValueError("ablation.variants must contain at least one variant.")
    return AblationProtocolSection(
        enabled=bool(loaded.get("enabled", False)),
        variants=variants,
    )


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Protocol field {key!r} must be a mapping.")
    return value


def load_staged_train_protocol(path: Path) -> StagedTrainProtocol:
    '''Load and validate a staged train protocol YAML file.

    Parameters
    ----------
    path : pathlib.Path
        Protocol YAML path on disk.

    Returns
    -------
    StagedTrainProtocol
        Parsed template ready for the train CLI and Optuna orchestration.

    Raises
    ------
    ValueError
        If required fields are missing or budgets fail production-claim checks.
    '''

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Protocol file {path} must contain a YAML mapping at the top level.")

    pdbbind_raw = _require_mapping(loaded, "pdbbind")
    dudez_raw = _require_mapping(loaded, "dudez")
    runtime_raw = _require_mapping(loaded, "runtime")
    reporting_raw = _require_mapping(loaded, "reporting")
    split_raw = _require_mapping(pdbbind_raw, "split")
    ablation = _load_ablation_section(loaded)

    production_claim: Optional[ProductionClaimRequirements] = None
    if "production_claim" in loaded:
        claim_raw = _require_mapping(loaded, "production_claim")
        production_claim = ProductionClaimRequirements(
            enforce=bool(claim_raw.get("enforce", False)),
            min_replicas=int(claim_raw["min_replicas"]),
            min_pdbbind_trials=int(claim_raw["min_pdbbind_trials"]),
            min_dudez_trials=int(claim_raw["min_dudez_trials"]),
        )

    protocol = StagedTrainProtocol(
        name=str(loaded["name"]),
        description=str(loaded.get("description", "")).strip(),
        replicas=int(loaded["replicas"]),
        seed=int(loaded["seed"]),
        pdbbind=PDBbindProtocolSection(
            target_column=str(pdbbind_raw.get("target_column", "experimental")),
            trials=int(pdbbind_raw["trials"]),
            epochs=int(pdbbind_raw["epochs"]),
            n_jobs=int(pdbbind_raw.get("n_jobs", 1)),
            search_phase=str(pdbbind_raw.get("search_phase", "full")),  # type: ignore[arg-type]
            enable_pruning=bool(pdbbind_raw.get("enable_pruning", True)),
            split_strategy=str(split_raw.get("strategy", "receptor_heldout")),
            split_train_size=float(split_raw.get("train_size", 0.6)),
            split_validation_size=float(split_raw.get("validation_size", 0.2)),
            split_test_size=float(split_raw.get("test_size", 0.2)),
        ),
        dudez=DUDEzProtocolSection(
            kind_column=str(dudez_raw.get("kind_column", "kind")),
            positive_kind=str(dudez_raw.get("positive_kind", "ligands")),
            negative_kind=str(dudez_raw.get("negative_kind", "decoys")),
            trials=int(dudez_raw["trials"]),
            epochs=int(dudez_raw["epochs"]),
            n_jobs=int(dudez_raw.get("n_jobs", 1)),
            primary_metric=str(dudez_raw.get("primary_metric", "BEDROC")),
            bedroc_alpha=float(dudez_raw.get("bedroc_alpha", 20.0)),
            scaling_strategy=str(dudez_raw.get("scaling_strategy", "pdbbind_scaler")),  # type: ignore[arg-type]
            ignore_unknown_kind=bool(dudez_raw.get("ignore_unknown_kind", False)),
        ),
        runtime=RuntimeProtocolSection(
            use_gpu=bool(runtime_raw.get("use_gpu", True)),
            pdbbind_only=bool(runtime_raw.get("pdbbind_only", False)),
            replica_jobs=int(runtime_raw.get("replica_jobs", 1)),
            resume_completed=bool(runtime_raw.get("resume_completed", False)),
        ),
        reporting=ReportingProtocolSection(
            generate_final_report=bool(reporting_raw.get("generate_final_report", False)),
            run_leakage_audit=bool(reporting_raw.get("run_leakage_audit", False)),
            run_baselines=bool(reporting_raw.get("run_baselines", False)),
            calibration_report_mode=str(
                reporting_raw.get("calibration_report_mode", "ranking_only")
            ),  # type: ignore[arg-type]
        ),
        ablation=ablation,
        source_path=path.resolve(),
        production_claim=production_claim,
        raw=loaded,
    )
    for label, value in (
        ("pdbbind.n_jobs", protocol.pdbbind.n_jobs),
        ("dudez.n_jobs", protocol.dudez.n_jobs),
        ("runtime.replica_jobs", protocol.runtime.replica_jobs),
    ):
        if int(value) < 1:
            raise ValueError(f"Protocol field {label} must be at least 1.")
    if protocol.dudez.bedroc_alpha <= 0.0:
        raise ValueError("Protocol field dudez.bedroc_alpha must be positive.")
    protocol.validate_production_claim_budget()
    return protocol


__all__ = [
    "ABLATION_VARIANT_ALIASES",
    "BUNDLED_PROTOCOL_DIR",
    "DEFAULT_ABLATION_VARIANTS",
    "AblationProtocolSection",
    "StagedTrainProtocol",
    "bundled_protocol_names",
    "load_staged_train_protocol",
    "normalize_ablation_variant_name",
    "resolve_protocol_path",
]
