# Description
###############################################################################
'''
Bundled multi-sample Snakefile for the "snakemake" vs_campaign execution
engine (see OCDocker.Workbench.VSDesign.plan_vs_campaign). Reads a
config-driven ``samples`` dict instead of a fixed ``input/{sample}/...``
directory layout, so discovered receptor/ligand/box files never need to be
relocated.

Config contract:
  samples: dict[str, dict] - sample name -> {receptor, ligand, box,
    row_kind ("vs" or "pipeline", default "vs"), engines (list),
    rescoring_engines (list, optional, "pipeline" rows only)}
  ocdocker_command: str - executable to invoke (default "ocdocker")
  common_args: list[str] - extra CLI flags appended to every sample's command
  results_dir: str - shared output directory (default "results")

Usage:

snakemake -s vs_campaign.smk --cores 4 --keep-going --rerun-incomplete \
  --config samples='{"s1": {"receptor": "r.pdb", "ligand": "l.smi", "box": "b.pdb", "engines": ["vina"]}}'
'''

# Imports
###############################################################################
import shlex


# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Rules
###############################################################################

SAMPLES = config.get("samples", {})
OCDOCKER_COMMAND = config.get("ocdocker_command", "ocdocker")
COMMON_ARGS = config.get("common_args", [])
RESULTS_DIR = config.get("results_dir", "results")


def _sample_command(sample):
    row = SAMPLES[sample]
    row_kind = row.get("row_kind", "vs")
    engines = row.get("engines") or ["vina"]
    command = [
        OCDOCKER_COMMAND, row_kind,
        "--receptor", row["receptor"],
        "--ligand", row["ligand"],
        "--box", row["box"],
        "--outdir", f"{RESULTS_DIR}/{sample}",
    ]
    if row_kind == "vs":
        command += ["--engine", engines[0]]
    else:
        command += ["--engines", ",".join(engines)]
        rescoring_engines = row.get("rescoring_engines")
        if rescoring_engines:
            command += ["--rescoring-engines", ",".join(rescoring_engines)]
    command += COMMON_ARGS
    return " ".join(shlex.quote(str(part)) for part in command)


rule all:
    input:
        expand(f"{RESULTS_DIR}/{{sample}}/.campaign_done", sample=list(SAMPLES.keys()))


rule ocdocker_campaign_sample:
    input:
        receptor=lambda wildcards: SAMPLES[wildcards.sample]["receptor"],
        ligand=lambda wildcards: SAMPLES[wildcards.sample]["ligand"],
        box=lambda wildcards: SAMPLES[wildcards.sample]["box"],
    output:
        f"{RESULTS_DIR}/{{sample}}/.campaign_done"
    params:
        command=lambda wildcards: _sample_command(wildcards.sample),
        outdir=lambda wildcards: f"{RESULTS_DIR}/{wildcards.sample}",
    resources:
        tmpdir=lambda wildcards: f"tmp/{wildcards.sample}",
    shell:
        "mkdir -p {params.outdir} && {params.command} && touch {output}"
