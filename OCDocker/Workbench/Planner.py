#!/usr/bin/env python3

# Description
###############################################################################
'''
Command-planning helpers for workbench specs.
'''

# Imports
###############################################################################
from __future__ import annotations

import json
from pathlib import Path

from OCDocker.Workbench.Models import FeaturePolicySelection
from OCDocker.Workbench.Models import OCScoreAblationSpec
from OCDocker.Workbench.Models import OCScoreInputSpec
from OCDocker.Workbench.Models import OCScoreStudySpec
from OCDocker.Workbench.Models import PlannedCommand
from OCDocker.Workbench.Models import RunManifest
from OCDocker.Workbench.Models import RunStatus
from OCDocker.Workbench.Models import SnakemakeWorkflowSpec
from OCDocker.Workbench.Models import VSCampaignSpec
from OCDocker.Workbench.Models import WorkbenchSpec

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


def _format_cli_value(value: object) -> str:
    '''Format a Snakemake config value for command-line use.

    Parameters
    ----------
    value : object
        Input value.

    Returns
    -------
    str
        Returned value.
    '''

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _extend_ocscore_input_args(command: list[str], inputs: OCScoreInputSpec) -> None:
    '''Append OCScore raw-input arguments to a command list.

    Parameters
    ----------
    command : list[str]
        Input value.
    inputs : OCScoreInputSpec
        Input value.
    '''

    if inputs.raw_input_dir is not None:
        command.extend(["--raw-input-dir", str(inputs.raw_input_dir)])
    elif inputs.merged_input is not None:
        command.extend(["--merged-input", str(inputs.merged_input)])
    else:
        command.extend(["--pdbbind-input", str(inputs.pdbbind_input)])
        command.extend(["--dudez-input", str(inputs.dudez_input)])


def _extend_feature_policy_args(
    command: list[str], selection: FeaturePolicySelection
) -> None:
    '''Append feature-policy selection arguments to a command list.

    Parameters
    ----------
    command : list[str]
        Input value.
    selection : FeaturePolicySelection
        Input value.
    '''

    for policy_dir in selection.policy_dirs:
        command.extend(["--feature-policy-dir", str(policy_dir)])
    for policy_yml in selection.policy_ymls:
        command.extend(["--feature-policy-yml", str(policy_yml)])
    for name in selection.names:
        command.extend(["--feature-policy", name])
    if selection.run_all:
        command.append("--run-all-feature-policies")


## Public ##


def plan_snakemake_command(
    workflow: SnakemakeWorkflowSpec, *, executable: str = "snakemake"
) -> PlannedCommand:
    '''Build a Snakemake command plan without executing the workflow.

    Parameters
    ----------
    workflow : SnakemakeWorkflowSpec
        Input value.
    executable : str
        Input value.

    Returns
    -------
    PlannedCommand
        Returned value.
    '''

    command = [
        executable,
        "-s",
        str(workflow.snakefile),
        "--cores",
        str(workflow.resources.cores),
    ]
    if workflow.profile is not None:
        command.extend(["--profile", str(workflow.profile)])
    if workflow.workdir is not None:
        command.extend(["--directory", str(workflow.workdir)])
    if workflow.use_conda:
        command.append("--use-conda")
    if workflow.keep_going:
        command.append("--keep-going")
    if workflow.rerun_incomplete:
        command.append("--rerun-incomplete")
    if workflow.dry_run:
        command.append("--dry-run")
    if workflow.config:
        command.append("--config")
        command.extend(
            f"{key}={_format_cli_value(value)}"
            for key, value in sorted(workflow.config.items())
        )
    command.extend(workflow.targets)

    return PlannedCommand(
        label="snakemake",
        command=tuple(command),
        cwd=workflow.workdir,
        writes=(),
    )


def plan_vs_campaign_command(
    spec: VSCampaignSpec, *, executable: str = "snakemake"
) -> PlannedCommand:
    '''Build a Snakemake command plan for a VS campaign spec.

    Parameters
    ----------
    spec : VSCampaignSpec
        Input value.
    executable : str
        Input value.

    Returns
    -------
    PlannedCommand
        Returned value.
    '''

    workflow = spec.workflow.model_copy(deep=True)
    if len(spec.inputs) == 1:
        input_spec = spec.inputs[0]
        workflow.config.setdefault("sample", input_spec.sample)
        workflow.config.setdefault("engines", ",".join(input_spec.engines))
        if input_spec.rescoring_engines:
            workflow.config.setdefault(
                "rescoring", ",".join(input_spec.rescoring_engines)
            )
    plan = plan_snakemake_command(workflow, executable=executable)
    return plan.model_copy(
        update={"label": f"vs_campaign:{spec.name}", "writes": (spec.workspace,)}
    )


def plan_ocscore_train_command(
    spec: OCScoreStudySpec | OCScoreAblationSpec,
    *,
    executable: str = "ocdocker",
) -> PlannedCommand:
    '''Build an ``ocdocker ocscore train`` command plan without running it.

    Parameters
    ----------
    spec : OCScoreStudySpec | OCScoreAblationSpec
        Input value.
    executable : str
        Input value.

    Returns
    -------
    PlannedCommand
        Returned value.
    '''

    command = [
        executable,
        "ocscore",
        "train",
        "--protocol",
        str(spec.protocol),
        "--output-dir",
        str(spec.output_dir),
    ]
    _extend_ocscore_input_args(command, spec.inputs)
    _extend_feature_policy_args(command, spec.feature_policies)
    return PlannedCommand(
        label=f"{spec.type}:{spec.name}",
        command=tuple(command),
        writes=(spec.output_dir,),
    )


def plan_command(
    spec: WorkbenchSpec,
    *,
    ocdocker_executable: str = "ocdocker",
    snakemake_executable: str = "snakemake",
) -> PlannedCommand:
    '''Dispatch command planning for any supported workbench spec.

    Parameters
    ----------
    spec : WorkbenchSpec
        Input value.
    ocdocker_executable : str
        Input value.
    snakemake_executable : str
        Input value.

    Returns
    -------
    PlannedCommand
        Returned value.
    '''

    if isinstance(spec, VSCampaignSpec):
        return plan_vs_campaign_command(spec, executable=snakemake_executable)
    if isinstance(spec, (OCScoreStudySpec, OCScoreAblationSpec)):
        return plan_ocscore_train_command(spec, executable=ocdocker_executable)
    raise TypeError(f"Unsupported workbench spec: {type(spec)!r}")


def build_run_manifest(
    spec: WorkbenchSpec,
    plan: PlannedCommand,
    *,
    run_id: str,
    status: RunStatus = "defined",
    workspace: str | Path | None = None,
) -> RunManifest:
    '''Create an initial run manifest from a spec and command plan.

    Parameters
    ----------
    spec : WorkbenchSpec
        Input value.
    plan : PlannedCommand
        Input value.
    run_id : str
        Input value.
    status : RunStatus
        Input value.
    workspace : str | Path | None
        Input value.

    Returns
    -------
    RunManifest
        Returned value.
    '''

    resolved_workspace = (
        Path(workspace) if workspace is not None else _default_workspace(spec)
    )
    return RunManifest(
        run_id=run_id,
        spec_type=spec.type,
        name=spec.name,
        status=status,
        workspace=resolved_workspace,
        command=plan.command,
    )


def _default_workspace(spec: WorkbenchSpec) -> Path:
    '''Return the default workspace path for a workbench spec.

    Parameters
    ----------
    spec : WorkbenchSpec
        Input value.

    Returns
    -------
    Path
        Returned value.
    '''

    if isinstance(spec, VSCampaignSpec):
        return spec.workspace
    return spec.output_dir


__all__ = [
    "build_run_manifest",
    "plan_command",
    "plan_ocscore_train_command",
    "plan_snakemake_command",
    "plan_vs_campaign_command",
]
