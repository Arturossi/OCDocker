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
