# Description
###############################################################################
'''
Snakemake example for running OCDocker as a scheduler-friendly docking job.

Usage:

snakemake -s examples/19_Snakefile_ocdocker_pipeline.smk --cores 4
'''

# Imports
###############################################################################
import os


# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Rules
###############################################################################


SAMPLE = "example"
OCDOCKER_COMMAND = config.get("ocdocker_command", "ocdocker")


rule all:
    input:
        f"results/{SAMPLE}/summary.json",
        f"results/{SAMPLE}/done.json",


rule ocdocker_pipeline:
    input:
        receptor="input/{sample}/receptor.pdbqt",
        ligand="input/{sample}/ligand.pdbqt",
        box="input/{sample}/box.txt",
    output:
        summary="results/{sample}/summary.json",
        done="results/{sample}/done.json",
    threads: 4
    conda:
        "envs/ocdocker.yml"
    params:
        outdir=lambda wildcards, output: os.path.dirname(output.summary),
        engines="vina,smina,plants",
        rescoring="oddt",
        command=OCDOCKER_COMMAND,
    resources:
        tmpdir=lambda wildcards: f"tmp/{wildcards.sample}",
    log:
        "logs/{sample}.log",
    shell:
        r'''
        mkdir -p "{resources.tmpdir}" "$(dirname "{log}")"
        OCDOCKER_THREADS="{threads}" \
        OCDOCKER_TMP_DIR="{resources.tmpdir}" \
        {params.command} \
          --threads "{threads}" \
          --tmp-dir "{resources.tmpdir}" \
          pipeline \
          --receptor "{input.receptor}" \
          --ligand "{input.ligand}" \
          --box "{input.box}" \
          --outdir "{params.outdir}" \
          --engines "{params.engines}" \
          --rescoring-engines "{params.rescoring}" \
          --strict-engines \
          --done-marker "{output.done}" \
          --log-file "{log}" \
          --no-stdout-log
        test -s "{output.summary}"
        test -s "{output.done}"
        '''
