#!/usr/bin/env python3

# Description
###############################################################################
"""
Declarative models for the OCDocker experiment workbench.
"""

# Imports
###############################################################################
from __future__ import annotations

from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

# License
###############################################################################
"""
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
"""

# Constants
###############################################################################


VALID_DOCKING_ENGINES = frozenset({"vina", "smina", "gnina", "plants"})
VALID_RESCORING_ENGINES = VALID_DOCKING_ENGINES | {"oddt"}

WorkbenchSpecType = Literal["vs_campaign", "ocscore_study", "ocscore_ablation"]
RunStatus = Literal["defined", "built", "dry_run", "running", "completed", "failed", "cancelled"]
PreflightSeverity = Literal["info", "warning", "error"]
MetricSortMode = Literal["min", "max"]
ComparisonDirection = Literal[
    "improved",
    "regressed",
    "unchanged",
    "incomplete",
]
WorkbenchPlotKind = Literal[
    "leaderboard_bar",
    "metric_scatter",
    "parallel_coordinates",
    "pareto_scatter",
]
WorkbenchReportFindingKind = Literal[
    "best_metric",
    "incomplete_metric",
    "missing_artifact",
    "no_results",
    "pareto_candidate",
    "pareto_skipped",
    "workspace_issue",
]
ArtifactKind = Literal[
    "json",
    "csv",
    "html",
    "markdown",
    "pdf",
    "image",
    "database",
    "log",
    "directory",
    "other",
]
EvidenceKind = Literal[
    "performance",
    "optimization",
    "shap",
    "figure",
    "prediction",
    "other",
]
OCScoreWorkspaceRole = Literal["baseline", "ablation"]
OCScoreExternalBaselineFamily = Literal[
    "scoring_function",
    "learned_sf",
    "sf_consensus",
    "descriptor_aggregate",
    "other",
]
OCScoreReplicaStatus = Literal["missing", "empty", "running", "completed", "failed", "unknown"]
OCScoreMetricDirection = Literal["max", "min"]


# Functions
###############################################################################
## Private ##


def _utc_now() -> datetime:
    """Return the current UTC timestamp for manifest metadata.

    Returns
    -------
    datetime
        Returned value.
    """

    return datetime.now(timezone.utc)


def _clean_string(value: str, field_name: str) -> str:
    """Return a stripped non-empty string or raise a validation error.

    Parameters
    ----------
    value : str
        Input value.
    field_name : str
        Input value.

    Returns
    -------
    str
        Returned value.
    """

    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty.")
    return cleaned


def _string_tuple(value: Any) -> tuple[str, ...]:
    """Normalize strings, comma-separated strings, or sequences to a string tuple.

    Parameters
    ----------
    value : Any
        Input value.

    Returns
    -------
    tuple[str, ...]
        Returned value.
    """

    if value is None:
        return ()
    if isinstance(value, str):
        items = value.split(",") if "," in value else [value]
    else:
        items = list(value)
    return tuple(_clean_string(str(item), "tuple item") for item in items if str(item).strip())


def _path_tuple(value: Any) -> tuple[Path, ...]:
    """Normalize path inputs to a tuple of Path objects.

    Parameters
    ----------
    value : Any
        Input value.

    Returns
    -------
    tuple[Path, ...]
        Returned value.
    """

    if value is None:
        return ()
    if isinstance(value, (str, Path)):
        items = [value]
    else:
        items = list(value)
    return tuple(Path(item) for item in items)


# Classes
###############################################################################


class WorkbenchModel(BaseModel):
    """Base model with strict fields for workbench schemas."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ResourceSpec(WorkbenchModel):
    """Runtime resources requested by a planned run.

    Attributes
    ----------
    cores : int
        Model field.
    memory_mb : int | None
        Model field.
    gpus : int
        Model field.
    """

    cores: int = Field(default=1, ge=1)
    memory_mb: int | None = Field(default=None, gt=0)
    gpus: int = Field(default=0, ge=0)


class SnakemakeWorkflowSpec(WorkbenchModel):
    """Snakemake workflow configuration used by VS campaign execution.

    Attributes
    ----------
    snakefile : Path
        Model field.
    workdir : Path | None
        Model field.
    profile : str | Path | None
        Model field.
    targets : tuple[str, ...]
        Model field.
    config : dict[str, Any]
        Model field.
    resources : ResourceSpec
        Model field.
    use_conda : bool
        Model field.
    keep_going : bool
        Model field.
    rerun_incomplete : bool
        Model field.
    dry_run : bool
        Model field.
    """

    snakefile: Path
    workdir: Path | None = None
    profile: str | Path | None = None
    targets: tuple[str, ...] = ()
    config: dict[str, Any] = Field(default_factory=dict)
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    use_conda: bool = True
    keep_going: bool = False
    rerun_incomplete: bool = True
    dry_run: bool = False

    @field_validator("targets", mode="before")
    @classmethod
    def _coerce_targets(cls, value: Any) -> tuple[str, ...]:
        """Coerce targets.

        Parameters
        ----------
        value : Any
            Input value.

        Returns
        -------
        tuple[str, ...]
            Returned value.
        """

        return _string_tuple(value)


class VSInputSpec(WorkbenchModel):
    """One receptor/ligand/box input set for a virtual-screening campaign.

    Attributes
    ----------
    sample : str
        Model field.
    receptor : Path
        Model field.
    ligand : Path
        Model field.
    box : Path
        Model field.
    engines : tuple[str, ...]
        Model field.
    rescoring_engines : tuple[str, ...] | None
        Model field.
    """

    sample: str
    receptor: Path
    ligand: Path
    box: Path
    engines: tuple[str, ...] = ("vina", "smina", "plants")
    rescoring_engines: tuple[str, ...] | None = None

    @field_validator("sample")
    @classmethod
    def _validate_sample(cls, value: str) -> str:
        """Validate sample.

        Parameters
        ----------
        value : str
            Input value.

        Returns
        -------
        str
            Returned value.
        """

        return _clean_string(value, "sample")

    @field_validator("engines", "rescoring_engines", mode="before")
    @classmethod
    def _coerce_engines(cls, value: Any) -> tuple[str, ...] | None:
        """Coerce engines.

        Parameters
        ----------
        value : Any
            Input value.

        Returns
        -------
        tuple[str, ...] | None
            Returned value.
        """

        if value is None:
            return None
        return tuple(engine.lower() for engine in _string_tuple(value))

    @model_validator(mode="after")
    def _validate_engines(self) -> Self:
        """Validate engines.

        Returns
        -------
        Self
            Returned value.
        """

        if not self.engines:
            raise ValueError("At least one docking engine is required.")
        invalid = sorted(set(self.engines) - VALID_DOCKING_ENGINES)
        if invalid:
            raise ValueError(f"Unsupported docking engine(s): {invalid}")
        if self.rescoring_engines is not None:
            invalid_rescoring = sorted(set(self.rescoring_engines) - VALID_RESCORING_ENGINES)
            if invalid_rescoring:
                raise ValueError(f"Unsupported rescoring engine(s): {invalid_rescoring}")
        return self


class VSCampaignSpec(WorkbenchModel):
    """High-level virtual-screening campaign definition.

    Attributes
    ----------
    schema_version : int
        Model field.
    type : Literal['vs_campaign']
        Model field.
    name : str
        Model field.
    workspace : Path
        Model field.
    workflow : SnakemakeWorkflowSpec
        Model field.
    inputs : tuple[VSInputSpec, ...]
        Model field.
    description : str
        Model field.
    tags : tuple[str, ...]
        Model field.
    """

    schema_version: int = 1
    type: Literal["vs_campaign"] = "vs_campaign"
    name: str
    workspace: Path
    workflow: SnakemakeWorkflowSpec
    inputs: tuple[VSInputSpec, ...]
    description: str = ""
    tags: tuple[str, ...] = ()

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """Validate name.

        Parameters
        ----------
        value : str
            Input value.

        Returns
        -------
        str
            Returned value.
        """

        return _clean_string(value, "name")

    @field_validator("inputs", mode="before")
    @classmethod
    def _coerce_inputs(cls, value: Any) -> Any:
        """Coerce inputs.

        Parameters
        ----------
        value : Any
            Input value.

        Returns
        -------
        Any
            Returned value.
        """

        if value is None:
            return ()
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags(cls, value: Any) -> tuple[str, ...]:
        """Coerce tags.

        Parameters
        ----------
        value : Any
            Input value.

        Returns
        -------
        tuple[str, ...]
            Returned value.
        """

        return _string_tuple(value)

    @model_validator(mode="after")
    def _validate_inputs(self) -> Self:
        """Validate inputs.

        Returns
        -------
        Self
            Returned value.
        """

        if not self.inputs:
            raise ValueError("A VS campaign requires at least one input set.")
        return self


class OCScoreInputSpec(WorkbenchModel):
    """Raw unreduced input selection for ``ocdocker ocscore train``.

    Attributes
    ----------
    raw_input_dir : Path | None
        Model field.
    merged_input : Path | None
        Model field.
    pdbbind_input : Path | None
        Model field.
    dudez_input : Path | None
        Model field.
    """

    raw_input_dir: Path | None = None
    merged_input: Path | None = None
    pdbbind_input: Path | None = None
    dudez_input: Path | None = None

    @model_validator(mode="after")
    def _validate_input_mode(self) -> Self:
        """Validate input mode.

        Returns
        -------
        Self
            Returned value.
        """

        raw = self.raw_input_dir is not None
        merged = self.merged_input is not None
        split_any = self.pdbbind_input is not None or self.dudez_input is not None
        split_complete = self.pdbbind_input is not None and self.dudez_input is not None

        if split_any and not split_complete:
            raise ValueError("pdbbind_input and dudez_input must be supplied together.")
        modes = sum(1 for enabled in (raw, merged, split_complete) if enabled)
        if modes != 1:
            raise ValueError(
                "Select exactly one OCScore input mode: raw_input_dir, merged_input, or pdbbind_input plus dudez_input."
            )
        return self


class FeaturePolicySelection(WorkbenchModel):
    """Feature-policy selection for OCScore optimization or ablation runs.

    Attributes
    ----------
    names : tuple[str, ...]
        Model field.
    policy_dirs : tuple[Path, ...]
        Model field.
    policy_ymls : tuple[Path, ...]
        Model field.
    run_all : bool
        Model field.
    """

    names: tuple[str, ...] = ()
    policy_dirs: tuple[Path, ...] = ()
    policy_ymls: tuple[Path, ...] = ()
    run_all: bool = False

    @field_validator("names", mode="before")
    @classmethod
    def _coerce_names(cls, value: Any) -> tuple[str, ...]:
        """Coerce names.

        Parameters
        ----------
        value : Any
            Input value.

        Returns
        -------
        tuple[str, ...]
            Returned value.
        """

        return _string_tuple(value)

    @field_validator("policy_dirs", "policy_ymls", mode="before")
    @classmethod
    def _coerce_paths(cls, value: Any) -> tuple[Path, ...]:
        """Coerce paths.

        Parameters
        ----------
        value : Any
            Input value.

        Returns
        -------
        tuple[Path, ...]
            Returned value.
        """

        return _path_tuple(value)

    @model_validator(mode="after")
    def _validate_selection(self) -> Self:
        """Validate selection.

        Returns
        -------
        Self
            Returned value.
        """

        if len(set(self.names)) != len(self.names):
            raise ValueError("Feature-policy names must be unique.")
        if self.run_all and self.names:
            raise ValueError("Use either run_all or explicit feature-policy names, not both.")
        return self


class OCScoreStudySpec(WorkbenchModel):
    """Single OCScore staged Optuna study definition.

    Attributes
    ----------
    schema_version : int
        Model field.
    type : Literal['ocscore_study']
        Model field.
    name : str
        Model field.
    protocol : str | Path
        Model field.
    inputs : OCScoreInputSpec
        Model field.
    output_dir : Path
        Model field.
    feature_policies : FeaturePolicySelection
        Model field.
    description : str
        Model field.
    tags : tuple[str, ...]
        Model field.
    """

    schema_version: int = 1
    type: Literal["ocscore_study"] = "ocscore_study"
    name: str
    protocol: str | Path
    inputs: OCScoreInputSpec
    output_dir: Path
    feature_policies: FeaturePolicySelection = Field(default_factory=FeaturePolicySelection)
    description: str = ""
    tags: tuple[str, ...] = ()

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """Validate name.

        Parameters
        ----------
        value : str
            Input value.

        Returns
        -------
        str
            Returned value.
        """

        return _clean_string(value, "name")

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags(cls, value: Any) -> tuple[str, ...]:
        """Coerce tags.

        Parameters
        ----------
        value : Any
            Input value.

        Returns
        -------
        tuple[str, ...]
            Returned value.
        """

        return _string_tuple(value)


class OCScoreAblationSpec(WorkbenchModel):
    """Feature-policy ablation process backed by ``ocdocker ocscore train``.

    Attributes
    ----------
    schema_version : int
        Model field.
    type : Literal['ocscore_ablation']
        Model field.
    name : str
        Model field.
    protocol : str | Path
        Model field.
    inputs : OCScoreInputSpec
        Model field.
    output_dir : Path
        Model field.
    feature_policies : FeaturePolicySelection
        Model field.
    include_full_reference : bool
        Model field.
    description : str
        Model field.
    tags : tuple[str, ...]
        Model field.
    """

    schema_version: int = 1
    type: Literal["ocscore_ablation"] = "ocscore_ablation"
    name: str
    protocol: str | Path
    inputs: OCScoreInputSpec
    output_dir: Path
    feature_policies: FeaturePolicySelection
    include_full_reference: bool = True
    description: str = ""
    tags: tuple[str, ...] = ()

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """Validate name.

        Parameters
        ----------
        value : str
            Input value.

        Returns
        -------
        str
            Returned value.
        """

        return _clean_string(value, "name")

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags(cls, value: Any) -> tuple[str, ...]:
        """Coerce tags.

        Parameters
        ----------
        value : Any
            Input value.

        Returns
        -------
        tuple[str, ...]
            Returned value.
        """

        return _string_tuple(value)

    @model_validator(mode="after")
    def _validate_ablation_scope(self) -> Self:
        """Validate ablation scope.

        Returns
        -------
        Self
            Returned value.
        """

        selection = self.feature_policies
        has_named_scope = bool(selection.names or selection.policy_ymls or selection.run_all)
        if not has_named_scope:
            raise ValueError("An ablation spec requires named policies, explicit policy YAMLs, or run_all.")
        return self


class PlannedCommand(WorkbenchModel):
    """Command plan generated from a workbench spec without executing it.

    Attributes
    ----------
    label : str
        Model field.
    command : tuple[str, ...]
        Model field.
    cwd : Path | None
        Model field.
    env : dict[str, str]
        Model field.
    writes : tuple[Path, ...]
        Model field.
    destructive : bool
        Model field.
    """

    label: str
    command: tuple[str, ...]
    cwd: Path | None = None
    env: dict[str, str] = Field(default_factory=dict)
    writes: tuple[Path, ...] = ()
    destructive: bool = False

    @field_validator("command", mode="before")
    @classmethod
    def _coerce_command(cls, value: Any) -> tuple[str, ...]:
        """Coerce command.

        Parameters
        ----------
        value : Any
            Input value.

        Returns
        -------
        tuple[str, ...]
            Returned value.
        """

        return _string_tuple(value)

    @field_validator("writes", mode="before")
    @classmethod
    def _coerce_writes(cls, value: Any) -> tuple[Path, ...]:
        """Coerce writes.

        Parameters
        ----------
        value : Any
            Input value.

        Returns
        -------
        tuple[Path, ...]
            Returned value.
        """

        return _path_tuple(value)


class PreflightCheck(WorkbenchModel):
    """One read-only preflight check for a Workbench spec."""

    code: str
    severity: PreflightSeverity
    passed: bool
    message: str
    path: Path | None = None
    subject: str = ""

    @field_validator("code", "message")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        """Validate non-empty string fields.

        Parameters
        ----------
        value : str
            Input value.

        Returns
        -------
        str
            Returned value.
        """

        return _clean_string(value, "preflight check field")


class PreflightReport(WorkbenchModel):
    """Read-only preflight report for a Workbench spec."""

    spec_path: Path | None = None
    spec_type: WorkbenchSpecType
    name: str
    ready: bool
    planned_command: tuple[str, ...]
    checks: tuple[PreflightCheck, ...] = ()
    error_count: int = Field(default=0, ge=0)
    warning_count: int = Field(default=0, ge=0)
    info_count: int = Field(default=0, ge=0)


class ResultArtifact(WorkbenchModel):
    """One result artifact recorded for a completed or partial run.

    Attributes
    ----------
    name : str
        Model field.
    path : Path
        Model field.
    kind : ArtifactKind
        Model field.
    role : str
        Model field.
    description : str
        Model field.
    """

    name: str
    path: Path
    kind: ArtifactKind = "other"
    role: str = ""
    description: str = ""

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """Validate name.

        Parameters
        ----------
        value : str
            Input value.

        Returns
        -------
        str
            Returned value.
        """

        return _clean_string(value, "name")


class RunManifest(WorkbenchModel):
    """Workbench run state intended for GUI, CLI, and future automation layers.

    Attributes
    ----------
    schema_version : int
        Model field.
    run_id : str
        Model field.
    spec_type : WorkbenchSpecType
        Model field.
    name : str
        Model field.
    status : RunStatus
        Model field.
    workspace : Path
        Model field.
    created_at : datetime
        Model field.
    updated_at : datetime
        Model field.
    command : tuple[str, ...]
        Model field.
    pid : int | None
        Model field.
    log_files : tuple[Path, ...]
        Model field.
    artifacts : tuple[ResultArtifact, ...]
        Model field.
    metadata : dict[str, Any]
        Model field.
    """

    schema_version: int = 1
    run_id: str
    spec_type: WorkbenchSpecType
    name: str
    status: RunStatus = "defined"
    workspace: Path
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    command: tuple[str, ...] = ()
    pid: int | None = None
    log_files: tuple[Path, ...] = ()
    artifacts: tuple[ResultArtifact, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id", "name")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        """Validate non empty.

        Parameters
        ----------
        value : str
            Input value.

        Returns
        -------
        str
            Returned value.
        """

        return _clean_string(value, "manifest field")

    @field_validator("command", mode="before")
    @classmethod
    def _coerce_command(cls, value: Any) -> tuple[str, ...]:
        """Coerce command.

        Parameters
        ----------
        value : Any
            Input value.

        Returns
        -------
        tuple[str, ...]
            Returned value.
        """

        return _string_tuple(value)

    @field_validator("log_files", mode="before")
    @classmethod
    def _coerce_log_files(cls, value: Any) -> tuple[Path, ...]:
        """Coerce log files.

        Parameters
        ----------
        value : Any
            Input value.

        Returns
        -------
        tuple[Path, ...]
            Returned value.
        """

        return _path_tuple(value)


class ResultManifest(WorkbenchModel):
    """Summary manifest for publishable and machine-readable run outputs.

    Attributes
    ----------
    schema_version : int
        Model field.
    run_id : str
        Model field.
    status : RunStatus
        Model field.
    artifacts : tuple[ResultArtifact, ...]
        Model field.
    metrics : dict[str, Any]
        Model field.
    generated_at : datetime
        Model field.
    """

    schema_version: int = 1
    run_id: str
    status: RunStatus
    artifacts: tuple[ResultArtifact, ...] = ()
    metrics: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=_utc_now)

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, value: str) -> str:
        """Validate run id.

        Parameters
        ----------
        value : str
            Input value.

        Returns
        -------
        str
            Returned value.
        """

        return _clean_string(value, "run_id")


class ExportedArtifact(WorkbenchModel):
    """Artifact entry prepared for a publishable Workbench export."""

    name: str
    source_path: Path
    export_path: Path | None = None
    kind: ArtifactKind = "other"
    role: str = ""
    description: str = ""
    exists: bool = False
    copied: bool = False


class PublicationExport(WorkbenchModel):
    """Publishable export scaffold generated from a Workbench manifest."""

    root: Path
    source_manifest_path: Path
    run_id: str
    status: RunStatus
    readme_path: Path
    publication_manifest_path: Path
    artifacts: tuple[ExportedArtifact, ...] = ()
    metrics: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=_utc_now)


class RunBundle(WorkbenchModel):
    """Prepared Workbench run bundle written without executing a command."""

    root: Path
    spec_path: Path
    plan_path: Path
    run_manifest_path: Path
    bundle_manifest_path: Path
    run_id: str
    spec_type: WorkbenchSpecType
    name: str
    command: tuple[str, ...]
    created_at: datetime = Field(default_factory=_utc_now)


class RunPathStatus(WorkbenchModel):
    """Filesystem status for one path referenced by a Workbench run."""

    path: Path
    exists: bool
    is_file: bool = False
    is_dir: bool = False
    name: str = ""
    role: str = ""


class ResultArtifactStatus(RunPathStatus):
    """Filesystem status plus metadata for one declared result artifact."""

    kind: ArtifactKind = "other"
    description: str = ""


class RunLogFilePreview(RunPathStatus):
    """Bounded text preview for one declared Workbench log file."""

    encoding: str = "utf-8"
    size_bytes: int = Field(default=0, ge=0)
    read_bytes: int = Field(default=0, ge=0)
    returned_line_count: int = Field(default=0, ge=0)
    truncated: bool = False
    lines: tuple[str, ...] = ()
    text: str = ""
    error: str = ""


class RunLogPreview(WorkbenchModel):
    """Bounded read-only log preview for one Workbench run manifest."""

    manifest_path: Path
    run_id: str
    spec_type: WorkbenchSpecType
    name: str
    status: RunStatus
    line_limit: int = Field(default=80, ge=1)
    byte_limit: int = Field(default=65536, ge=1)
    encoding: str = "utf-8"
    logs: tuple[RunLogFilePreview, ...] = ()


class RunLaunchPlan(WorkbenchModel):
    """Non-executing launch envelope for a prepared Workbench run."""

    manifest_path: Path
    run_id: str
    spec_type: WorkbenchSpecType
    name: str
    status: RunStatus
    workspace: Path
    cwd: Path
    command: tuple[str, ...]
    shell_command: str
    foreground_command: str
    background_command: str
    log_dir: Path
    stdout_log: Path
    stderr_log: Path
    pid_file: Path
    script_path: Path | None = None
    script_written: bool = False


class RunStatusReport(WorkbenchModel):
    """Read-only status report for one Workbench run manifest."""

    manifest_path: Path
    run_id: str
    spec_type: WorkbenchSpecType
    name: str
    status: RunStatus
    workspace: Path
    workspace_status: RunPathStatus
    updated_at: datetime
    command: tuple[str, ...] = ()
    pid: int | None = None
    pid_alive: bool | None = None
    result_manifest_path: Path | None = None
    result_manifest_exists: bool = False
    log_files: tuple[RunPathStatus, ...] = ()
    artifacts: tuple[RunPathStatus, ...] = ()


class ResultSummary(WorkbenchModel):
    """Read-only summary of artifacts and metrics declared by a manifest."""

    source_manifest_path: Path
    source_type: Literal["run_manifest", "result_manifest"]
    run_id: str
    status: RunStatus
    generated_at: datetime | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: tuple[ResultArtifactStatus, ...] = ()
    artifact_count: int = Field(default=0, ge=0)
    existing_artifact_count: int = Field(default=0, ge=0)
    missing_artifact_count: int = Field(default=0, ge=0)


class MetricLeaderboardEntry(WorkbenchModel):
    """One result-manifest row in a metric leaderboard."""

    manifest_path: Path
    run_id: str
    status: RunStatus
    metric_name: str
    metric_value: float | None = None
    rank: int | None = Field(default=None, ge=1)
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifact_count: int = Field(default=0, ge=0)
    missing_artifact_count: int = Field(default=0, ge=0)
    included: bool = False
    exclusion_reason: str = ""


class InventoryIssue(WorkbenchModel):
    """Non-fatal issue found while scanning a Workbench root."""

    path: Path
    message: str

    @field_validator("message")
    @classmethod
    def _validate_message(cls, value: str) -> str:
        """Validate message.

        Parameters
        ----------
        value : str
            Input value.

        Returns
        -------
        str
            Returned value.
        """

        return _clean_string(value, "message")


class WorkbenchAdoptionCandidate(WorkbenchModel):
    """One existing output directory that can be adopted into Workbench."""

    source_path: Path
    run_id: str
    spec_type: WorkbenchSpecType
    name: str
    status: RunStatus
    workspace: Path
    metric_files: tuple[Path, ...] = ()
    log_files: tuple[Path, ...] = ()
    artifacts: tuple[ResultArtifact, ...] = ()
    metrics: dict[str, Any] = Field(default_factory=dict)
    issue_count: int = Field(default=0, ge=0)
    issues: tuple[InventoryIssue, ...] = ()


class WorkbenchAdoptionPlan(WorkbenchModel):
    """Dry-run adoption plan for existing OCDocker output directories."""

    source_root: Path
    max_depth: int = Field(default=3, ge=0)
    scanned_at: datetime = Field(default_factory=_utc_now)
    spec_type: WorkbenchSpecType
    status: RunStatus | None = None
    run_id_prefix: str = ""
    require_metrics: bool = False
    candidate_count: int = Field(default=0, ge=0)
    candidates: tuple[WorkbenchAdoptionCandidate, ...] = ()
    issue_count: int = Field(default=0, ge=0)
    issues: tuple[InventoryIssue, ...] = ()


class WorkbenchAdoptedRun(WorkbenchModel):
    """One run manifest pair written by an adoption operation."""

    source_path: Path
    run_id: str
    workspace: Path
    run_manifest_path: Path
    result_manifest_path: Path
    metric_count: int = Field(default=0, ge=0)
    artifact_count: int = Field(default=0, ge=0)
    log_count: int = Field(default=0, ge=0)


class WorkbenchAdoptionResult(WorkbenchModel):
    """Summary of manifests written for adopted existing runs."""

    source_root: Path
    destination_root: Path
    created_at: datetime = Field(default_factory=_utc_now)
    run_count: int = Field(default=0, ge=0)
    runs: tuple[WorkbenchAdoptedRun, ...] = ()
    issue_count: int = Field(default=0, ge=0)
    issues: tuple[InventoryIssue, ...] = ()


class RunDetail(WorkbenchModel):
    """Aggregate read-only drill-down for one Workbench run."""

    target: Path
    manifest_path: Path
    run_id: str
    spec_type: WorkbenchSpecType
    name: str
    status: RunStatus
    status_report: RunStatusReport
    log_preview: RunLogPreview | None = None
    result_summary: ResultSummary | None = None
    issue_count: int = Field(default=0, ge=0)
    issues: tuple[InventoryIssue, ...] = ()


class MetricCatalogEntry(WorkbenchModel):
    """Coverage and numeric summary for one discovered metric."""

    metric_name: str
    observed_count: int = Field(default=0, ge=0)
    numeric_count: int = Field(default=0, ge=0)
    non_numeric_count: int = Field(default=0, ge=0)
    missing_count: int = Field(default=0, ge=0)
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None


class MetricCatalog(WorkbenchModel):
    """Read-only coverage catalog for metrics across result manifests."""

    root: Path
    max_depth: int = Field(default=6, ge=0)
    scanned_at: datetime = Field(default_factory=_utc_now)
    result_manifest_count: int = Field(default=0, ge=0)
    metric_count: int = Field(default=0, ge=0)
    metrics: tuple[MetricCatalogEntry, ...] = ()
    issue_count: int = Field(default=0, ge=0)
    issues: tuple[InventoryIssue, ...] = ()


class ParetoObjective(WorkbenchModel):
    """One metric objective used to build a Pareto front."""

    metric_name: str
    mode: MetricSortMode = "max"


class ParetoEntry(WorkbenchModel):
    """One result-manifest entry considered for a Pareto front."""

    manifest_path: Path
    run_id: str
    status: RunStatus
    metric_values: dict[str, float] = Field(default_factory=dict)
    missing_metrics: tuple[str, ...] = ()
    non_numeric_metrics: tuple[str, ...] = ()
    dominated_by: tuple[str, ...] = ()
    included: bool = False


class ParetoFront(WorkbenchModel):
    """Read-only multi-objective Pareto-front summary."""

    root: Path
    max_depth: int = Field(default=6, ge=0)
    scanned_at: datetime = Field(default_factory=_utc_now)
    objectives: tuple[ParetoObjective, ...]
    front_entries: tuple[ParetoEntry, ...] = ()
    dominated_entries: tuple[ParetoEntry, ...] = ()
    skipped_entries: tuple[ParetoEntry, ...] = ()
    result_manifest_count: int = Field(default=0, ge=0)
    issue_count: int = Field(default=0, ge=0)
    issues: tuple[InventoryIssue, ...] = ()


class MetricLeaderboard(WorkbenchModel):
    """Read-only ranking of result manifests by one numeric metric."""

    root: Path
    metric_name: str
    mode: MetricSortMode = "max"
    max_depth: int = Field(default=6, ge=0)
    scanned_at: datetime = Field(default_factory=_utc_now)
    ranked_entries: tuple[MetricLeaderboardEntry, ...] = ()
    skipped_entries: tuple[MetricLeaderboardEntry, ...] = ()
    best_entry: MetricLeaderboardEntry | None = None
    issue_count: int = Field(default=0, ge=0)
    issues: tuple[InventoryIssue, ...] = ()


class MetricMatrixRow(WorkbenchModel):
    """One result-manifest row in a metric matrix."""

    manifest_path: Path
    run_id: str
    status: RunStatus
    metric_values: dict[str, float] = Field(default_factory=dict)
    raw_metrics: dict[str, Any] = Field(default_factory=dict)
    missing_metrics: tuple[str, ...] = ()
    non_numeric_metrics: tuple[str, ...] = ()
    artifact_count: int = Field(default=0, ge=0)
    missing_artifact_count: int = Field(default=0, ge=0)


class MetricMatrix(WorkbenchModel):
    """Read-only matrix of numeric metrics across result manifests."""

    root: Path
    max_depth: int = Field(default=6, ge=0)
    scanned_at: datetime = Field(default_factory=_utc_now)
    metric_names: tuple[str, ...] = ()
    rows: tuple[MetricMatrixRow, ...] = ()
    result_manifest_count: int = Field(default=0, ge=0)
    issue_count: int = Field(default=0, ge=0)
    issues: tuple[InventoryIssue, ...] = ()


class RunInventoryItem(WorkbenchModel):
    """Compact summary of one discovered run manifest."""

    manifest_path: Path
    run_id: str
    spec_type: WorkbenchSpecType
    name: str
    status: RunStatus
    workspace: Path
    updated_at: datetime
    artifact_count: int = Field(default=0, ge=0)
    missing_artifacts: tuple[Path, ...] = ()


class WorkbenchComparisonMetric(WorkbenchModel):
    """One metric delta between a baseline and candidate run."""

    metric_name: str
    mode: MetricSortMode = "max"
    baseline_value: float | None = None
    candidate_value: float | None = None
    delta: float | None = None
    percent_delta: float | None = None
    direction: ComparisonDirection = "incomplete"
    improved: bool = False
    regressed: bool = False
    baseline_missing: bool = False
    candidate_missing: bool = False
    baseline_non_numeric: bool = False
    candidate_non_numeric: bool = False


class WorkbenchComparisonCandidate(WorkbenchModel):
    """Comparison summary for one candidate run against a baseline."""

    run_id: str
    status: RunStatus
    manifest_path: Path
    metrics: tuple[WorkbenchComparisonMetric, ...] = ()
    improved_count: int = Field(default=0, ge=0)
    regressed_count: int = Field(default=0, ge=0)
    unchanged_count: int = Field(default=0, ge=0)
    incomplete_count: int = Field(default=0, ge=0)
    net_score: int = 0
    artifact_count: int = Field(default=0, ge=0)
    missing_artifact_count: int = Field(default=0, ge=0)


class WorkbenchComparison(WorkbenchModel):
    """Read-only comparison of candidate result manifests against a baseline."""

    root: Path
    max_depth: int = Field(default=6, ge=0)
    scanned_at: datetime = Field(default_factory=_utc_now)
    baseline_run_id: str
    baseline_manifest_path: Path
    baseline_status: RunStatus
    baseline_artifact_count: int = Field(default=0, ge=0)
    baseline_missing_artifact_count: int = Field(default=0, ge=0)
    metrics: tuple[ParetoObjective, ...] = ()
    result_manifest_count: int = Field(default=0, ge=0)
    candidate_count: int = Field(default=0, ge=0)
    candidates: tuple[WorkbenchComparisonCandidate, ...] = ()
    best_candidate: WorkbenchComparisonCandidate | None = None
    issue_count: int = Field(default=0, ge=0)
    issues: tuple[InventoryIssue, ...] = ()


class WorkbenchAblationCandidate(WorkbenchModel):
    """One ablation policy compared against a reference run."""

    policy_name: str
    run_id: str
    status: RunStatus
    manifest_path: Path
    source_path: Path | None = None
    metrics: tuple[WorkbenchComparisonMetric, ...] = ()
    improved_count: int = Field(default=0, ge=0)
    regressed_count: int = Field(default=0, ge=0)
    unchanged_count: int = Field(default=0, ge=0)
    incomplete_count: int = Field(default=0, ge=0)
    net_score: int = 0
    artifact_count: int = Field(default=0, ge=0)
    missing_artifact_count: int = Field(default=0, ge=0)


class WorkbenchAblationAnalysis(WorkbenchModel):
    """Read-only OCScore ablation comparison against a reference run."""

    root: Path
    max_depth: int = Field(default=6, ge=0)
    scanned_at: datetime = Field(default_factory=_utc_now)
    baseline_run_id: str
    baseline_policy_name: str
    baseline_manifest_path: Path
    baseline_source_path: Path | None = None
    metrics: tuple[ParetoObjective, ...] = ()
    result_manifest_count: int = Field(default=0, ge=0)
    detected_ablation_count: int = Field(default=0, ge=0)
    candidate_count: int = Field(default=0, ge=0)
    candidates: tuple[WorkbenchAblationCandidate, ...] = ()
    best_candidate: WorkbenchAblationCandidate | None = None
    issue_count: int = Field(default=0, ge=0)
    issues: tuple[InventoryIssue, ...] = ()


class WorkbenchArtifactEntry(WorkbenchModel):
    """One artifact row in a cross-run Workbench artifact index."""

    source_type: Literal["run_manifest", "result_manifest"]
    source_manifest_path: Path
    run_id: str
    status: RunStatus
    name: str
    path: Path
    kind: ArtifactKind = "other"
    role: str = ""
    description: str = ""
    exists: bool = False
    is_file: bool = False
    is_dir: bool = False
    suffix: str = ""
    size_bytes: int | None = Field(default=None, ge=0)
    modified_at: datetime | None = None


class WorkbenchArtifactIndex(WorkbenchModel):
    """Read-only cross-run index of declared Workbench artifacts."""

    root: Path
    max_depth: int = Field(default=6, ge=0)
    scanned_at: datetime = Field(default_factory=_utc_now)
    filters: dict[str, Any] = Field(default_factory=dict)
    run_manifest_count: int = Field(default=0, ge=0)
    result_manifest_count: int = Field(default=0, ge=0)
    artifact_count: int = Field(default=0, ge=0)
    existing_artifact_count: int = Field(default=0, ge=0)
    missing_artifact_count: int = Field(default=0, ge=0)
    kind_counts: dict[str, int] = Field(default_factory=dict)
    role_counts: dict[str, int] = Field(default_factory=dict)
    entries: tuple[WorkbenchArtifactEntry, ...] = ()
    issue_count: int = Field(default=0, ge=0)
    issues: tuple[InventoryIssue, ...] = ()


class WorkbenchEvidenceEntry(WorkbenchModel):
    """One discovered OCScore evidence artifact or table."""

    run_id: str
    status: RunStatus
    manifest_path: Path
    source_path: Path | None = None
    path: Path
    kind: EvidenceKind = "other"
    role: str = ""
    dataset: str = ""
    policy_name: str = ""
    replica: str = ""
    figure_name: str = ""
    comparison_key: str = ""
    suffix: str = ""
    size_bytes: int | None = Field(default=None, ge=0)
    modified_at: datetime | None = None
    column_count: int | None = Field(default=None, ge=0)
    metric_names: tuple[str, ...] = ()


class WorkbenchEvidenceIndex(WorkbenchModel):
    """Read-only index of OCScore evidence discovered from adopted sources."""

    root: Path
    max_depth: int = Field(default=6, ge=0)
    source_depth: int = Field(default=6, ge=0)
    scanned_at: datetime = Field(default_factory=_utc_now)
    result_manifest_count: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    kind_counts: dict[str, int] = Field(default_factory=dict)
    role_counts: dict[str, int] = Field(default_factory=dict)
    entries: tuple[WorkbenchEvidenceEntry, ...] = ()
    performance_points: tuple[dict[str, Any], ...] = ()
    optimization_points: tuple[dict[str, Any], ...] = ()
    shap_features: tuple[dict[str, Any], ...] = ()
    issue_count: int = Field(default=0, ge=0)
    issues: tuple[InventoryIssue, ...] = ()


class WorkbenchOCScoreMetric(WorkbenchModel):
    """One curated OCScore metric value for a replica."""

    name: str
    label: str = ""
    direction: OCScoreMetricDirection = "max"
    value: float
    observation_count: int = Field(default=1, ge=1)
    source_paths: tuple[Path, ...] = ()


class WorkbenchOCScoreFigure(WorkbenchModel):
    """One figure discovered for a strict OCScore study or replica."""

    path: Path
    role: str = "figure"
    dataset: str = ""
    metric_name: str = ""
    policy_name: str = ""
    replica_name: str = ""
    suffix: str = ""
    size_bytes: int | None = Field(default=None, ge=0)
    modified_at: datetime | None = None


class WorkbenchOCScoreReplica(WorkbenchModel):
    """One baseline or ablation replica in the strict OCScore layout."""

    role: OCScoreWorkspaceRole
    study_name: str
    policy_name: str
    replica_name: str
    replica_index: int = Field(ge=1)
    path: Path
    exists: bool = False
    status: OCScoreReplicaStatus = "missing"
    metrics: tuple[WorkbenchOCScoreMetric, ...] = ()
    figures: tuple[WorkbenchOCScoreFigure, ...] = ()
    log_files: tuple[Path, ...] = ()
    issues: tuple[str, ...] = ()


class WorkbenchOCScoreCrossValidationMetric(WorkbenchModel):
    """One mean/std row from exported cross-validation scorer summaries."""

    scorer: str
    metric: str
    mean: float
    std: float = 0.0
    n_folds: int = Field(default=0, ge=0)


class WorkbenchOCScoreCrossValidation(WorkbenchModel):
    """Cross-validation summary discovered under a study export directory."""

    path: Path
    task: str = ""
    fold_count: int = Field(default=0, ge=0)
    metrics: tuple[WorkbenchOCScoreCrossValidationMetric, ...] = ()


class WorkbenchOCScoreStudy(WorkbenchModel):
    """A strict OCScore baseline or ablation study summary."""

    role: OCScoreWorkspaceRole
    study_name: str
    policy_name: str
    path: Path
    expected_replica_count: int = Field(default=5, ge=1)
    detected_replica_count: int = Field(default=0, ge=0)
    completed_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    missing_count: int = Field(default=0, ge=0)
    replicas: tuple[WorkbenchOCScoreReplica, ...] = ()
    figures: tuple[WorkbenchOCScoreFigure, ...] = ()
    metric_summary: dict[str, dict[str, Any]] = Field(default_factory=dict)
    cross_validation: WorkbenchOCScoreCrossValidation | None = None


class WorkbenchOCScoreExternalBaseline(WorkbenchModel):
    """One external baseline row from baselines_summary.csv."""

    baseline_name: str
    baseline_family: OCScoreExternalBaselineFamily | str
    split: str
    path: Path
    metric_summary: dict[str, dict[str, Any]] = Field(default_factory=dict)
    n_replicas: int = Field(default=0, ge=0)


class WorkbenchOCScoreWorkspace(WorkbenchModel):
    """Strict OCScore workspace payload used by the Workbench dashboard."""

    root: Path
    scanned_at: datetime = Field(default_factory=_utc_now)
    expected_replica_count: int = Field(default=5, ge=1)
    max_depth: int = Field(default=6, ge=0)
    baseline_study: WorkbenchOCScoreStudy
    ablation_studies: tuple[WorkbenchOCScoreStudy, ...] = ()
    external_baselines: tuple[WorkbenchOCScoreExternalBaseline, ...] = ()
    study_count: int = Field(default=0, ge=0)
    replica_count: int = Field(default=0, ge=0)
    completed_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    missing_count: int = Field(default=0, ge=0)
    metric_names: tuple[str, ...] = ()
    issue_count: int = Field(default=0, ge=0)
    issues: tuple[InventoryIssue, ...] = ()


class WorkspaceInventory(WorkbenchModel):
    """Read-only inventory of Workbench manifests below a root path."""

    root: Path
    max_depth: int = Field(default=6, ge=0)
    scanned_at: datetime = Field(default_factory=_utc_now)
    runs: tuple[RunInventoryItem, ...] = ()
    result_manifests: tuple[Path, ...] = ()
    issues: tuple[InventoryIssue, ...] = ()


class WorkspaceOverview(WorkbenchModel):
    """Read-only dashboard overview of a Workbench workspace."""

    root: Path
    max_depth: int = Field(default=6, ge=0)
    scanned_at: datetime = Field(default_factory=_utc_now)
    run_count: int = Field(default=0, ge=0)
    result_manifest_count: int = Field(default=0, ge=0)
    issue_count: int = Field(default=0, ge=0)
    missing_artifact_count: int = Field(default=0, ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)
    spec_type_counts: dict[str, int] = Field(default_factory=dict)
    recent_runs: tuple[RunInventoryItem, ...] = ()
    issues: tuple[InventoryIssue, ...] = ()


class WorkbenchPlot(WorkbenchModel):
    """Plot-ready payload for GUI and notebook rendering."""

    root: Path
    plot_kind: WorkbenchPlotKind
    title: str
    metric_names: tuple[str, ...] = ()
    data: tuple[dict[str, Any], ...] = ()
    layout: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    included_count: int = Field(default=0, ge=0)
    skipped_count: int = Field(default=0, ge=0)
    issue_count: int = Field(default=0, ge=0)
    issues: tuple[InventoryIssue, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkbenchReportFinding(WorkbenchModel):
    """One decision-support finding in a Workbench analysis report."""

    kind: WorkbenchReportFindingKind
    severity: PreflightSeverity = "info"
    title: str
    message: str
    run_id: str = ""
    metric_name: str = ""
    metric_value: float | None = None
    manifest_path: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkbenchAnalysisReport(WorkbenchModel):
    """Composed read-only analysis report for GUI and publication workflows."""

    root: Path
    max_depth: int = Field(default=6, ge=0)
    recent_limit: int = Field(default=20, ge=1)
    top_n: int = Field(default=5, ge=1)
    scanned_at: datetime = Field(default_factory=_utc_now)
    overview: WorkspaceOverview
    metrics_catalog: MetricCatalog
    metric_matrix: MetricMatrix | None = None
    leaderboards: tuple[MetricLeaderboard, ...] = ()
    pareto_front: ParetoFront | None = None
    findings: tuple[WorkbenchReportFinding, ...] = ()
    issue_count: int = Field(default=0, ge=0)
    markdown: str = ""


WorkbenchSpec = VSCampaignSpec | OCScoreStudySpec | OCScoreAblationSpec


__all__ = [
    "ArtifactKind",
    "ComparisonDirection",
    "ExportedArtifact",
    "FeaturePolicySelection",
    "InventoryIssue",
    "MetricSortMode",
    "MetricLeaderboardEntry",
    "MetricMatrixRow",
    "MetricMatrix",
    "MetricLeaderboard",
    "ParetoFront",
    "ParetoEntry",
    "ParetoObjective",
    "MetricCatalog",
    "MetricCatalogEntry",
    "OCScoreAblationSpec",
    "WorkbenchOCScoreWorkspace",
    "WorkbenchOCScoreStudy",
    "WorkbenchOCScoreReplica",
    "WorkbenchOCScoreMetric",
    "WorkbenchOCScoreFigure",
    "OCScoreWorkspaceRole",
    "OCScoreReplicaStatus",
    "OCScoreMetricDirection",
    "OCScoreInputSpec",
    "OCScoreStudySpec",
    "PreflightCheck",
    "PreflightReport",
    "PreflightSeverity",
    "PlannedCommand",
    "PublicationExport",
    "ResourceSpec",
    "ResultArtifact",
    "ResultArtifactStatus",
    "ResultManifest",
    "ResultSummary",
    "RunBundle",
    "RunInventoryItem",
    "RunLogFilePreview",
    "RunLogPreview",
    "RunLaunchPlan",
    "RunManifest",
    "RunPathStatus",
    "RunStatus",
    "RunStatusReport",
    "SnakemakeWorkflowSpec",
    "VALID_DOCKING_ENGINES",
    "VALID_RESCORING_ENGINES",
    "VSInputSpec",
    "VSCampaignSpec",
    "WorkbenchAblationAnalysis",
    "WorkbenchAblationCandidate",
    "WorkbenchAnalysisReport",
    "EvidenceKind",
    "WorkbenchArtifactEntry",
    "WorkbenchArtifactIndex",
    "WorkbenchEvidenceEntry",
    "WorkbenchEvidenceIndex",
    "WorkbenchComparison",
    "WorkbenchComparisonCandidate",
    "WorkbenchComparisonMetric",
    "WorkbenchPlot",
    "WorkbenchPlotKind",
    "WorkspaceInventory",
    "WorkspaceOverview",
    "WorkbenchModel",
    "WorkbenchReportFinding",
    "WorkbenchReportFindingKind",
    "WorkbenchSpec",
    "WorkbenchSpecType",
]
