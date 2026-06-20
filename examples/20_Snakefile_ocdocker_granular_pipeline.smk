# Description
###############################################################################
'''
Granular Snakemake example for running OCDocker pipeline stages.

Usage:

snakemake -s examples/20_Snakefile_ocdocker_granular_pipeline.smk --cores 12
snakemake -s examples/20_Snakefile_ocdocker_granular_pipeline.smk --cores 12 --config ocdocker_command=ocd
'''

# Imports
###############################################################################
import os


# License
###############################################################################
'''OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Rules
###############################################################################


SAMPLE = config.get("sample", "example")
OCDOCKER_COMMAND = config.get("ocdocker_command", "ocdocker")
DEFAULT_THREADS = int(config.get("threads", 4))
DEFAULT_TIMEOUT = int(config.get("timeout", 900))
RAW_ENGINES = config.get("engines", ["vina", "smina", "plants"])
ENGINES = [engine.strip() for engine in RAW_ENGINES.split(",")] if isinstance(RAW_ENGINES, str) else list(RAW_ENGINES)
RESCORING_ENGINES = config.get("rescoring", ",".join(ENGINES))


rule all:
    input:
        f"results/granular/{SAMPLE}/summary.json",


rule pipeline_prepare:
    input:
        receptor="input/{sample}/receptor.pdbqt",
        ligand="input/{sample}/ligand.pdbqt",
        box="input/{sample}/box.txt",
    output:
        manifest="results/granular/{sample}/prepare_manifest.json",
        done="results/granular/{sample}/prepare.done.json",
    threads: 1
    conda:
        "envs/ocdocker.yml"
    params:
        command=OCDOCKER_COMMAND,
        outdir=lambda wildcards, output: os.path.dirname(output.manifest),
        engines=",".join(ENGINES),
    resources:
        tmpdir=lambda wildcards: f"tmp/granular/{wildcards.sample}/prepare",
    log:
        "logs/granular/{sample}/prepare.log",
    shell:
        r'''
        mkdir -p "{resources.tmpdir}" "$(dirname "{log}")"
        OCDOCKER_THREADS="{threads}"         OCDOCKER_TMP_DIR="{resources.tmpdir}"         {params.command}           --threads "{threads}"           --tmp-dir "{resources.tmpdir}"           pipeline prepare           --receptor "{input.receptor}"           --ligand "{input.ligand}"           --box "{input.box}"           --outdir "{params.outdir}"           --engines "{params.engines}"           --done-marker "{output.done}"           --log-file "{log}"           --no-stdout-log
        test -s "{output.manifest}"
        test -s "{output.done}"
        '''


rule pipeline_dock_engine:
    input:
        prepared=rules.pipeline_prepare.output.manifest,
        receptor="input/{sample}/receptor.pdbqt",
        ligand="input/{sample}/ligand.pdbqt",
        box="input/{sample}/box.txt",
    output:
        manifest="results/granular/{sample}/{engine}Files/dock_manifest.json",
        done="results/granular/{sample}/dock_{engine}.done.json",
    threads: DEFAULT_THREADS
    conda:
        "envs/ocdocker.yml"
    params:
        command=OCDOCKER_COMMAND,
        outdir=lambda wildcards, output: str(os.path.dirname(os.path.dirname(output.manifest))),
        timeout=DEFAULT_TIMEOUT,
    resources:
        tmpdir=lambda wildcards: f"tmp/granular/{wildcards.sample}/dock/{wildcards.engine}",
        gpu=lambda wildcards: 1 if wildcards.engine == "gnina" else 0,
    log:
        "logs/granular/{sample}/dock_{engine}.log",
    shell:
        r'''
        mkdir -p "{resources.tmpdir}" "$(dirname "{log}")"
        OCDOCKER_THREADS="{threads}"         OCDOCKER_TMP_DIR="{resources.tmpdir}"         {params.command}           --threads "{threads}"           --tmp-dir "{resources.tmpdir}"           pipeline dock           --receptor "{input.receptor}"           --ligand "{input.ligand}"           --box "{input.box}"           --outdir "{params.outdir}"           --engines "{wildcards.engine}"           --strict-engines           --done-marker "{output.done}"           --timeout "{params.timeout}"           --log-file "{log}"           --no-stdout-log
        test -s "{output.manifest}"
        test -s "{output.done}"
        '''


rule pipeline_collect:
    input:
        manifests=expand("results/granular/{sample}/{engine}Files/dock_manifest.json", sample=SAMPLE, engine=ENGINES),
    output:
        inventory="results/granular/{sample}/pose_inventory.csv",
        manifest="results/granular/{sample}/collect_manifest.json",
        done="results/granular/{sample}/collect.done.json",
    threads: 1
    conda:
        "envs/ocdocker.yml"
    params:
        command=OCDOCKER_COMMAND,
        outdir=lambda wildcards, output: os.path.dirname(output.inventory),
    resources:
        tmpdir=lambda wildcards: f"tmp/granular/{wildcards.sample}/collect",
    log:
        "logs/granular/{sample}/collect.log",
    shell:
        r'''
        mkdir -p "{resources.tmpdir}" "$(dirname "{log}")"
        {params.command}           --threads "{threads}"           --tmp-dir "{resources.tmpdir}"           pipeline collect           --outdir "{params.outdir}"           --done-marker "{output.done}"           --log-file "{log}"           --no-stdout-log
        test -s "{output.inventory}"
        test -s "{output.manifest}"
        test -s "{output.done}"
        '''


rule pipeline_cluster:
    input:
        inventory=rules.pipeline_collect.output.inventory,
    output:
        manifest="results/granular/{sample}/cluster_manifest.json",
        representative="results/granular/{sample}/representative.mol2",
        done="results/granular/{sample}/cluster.done.json",
    threads: DEFAULT_THREADS
    conda:
        "envs/ocdocker.yml"
    params:
        command=OCDOCKER_COMMAND,
        outdir=lambda wildcards, output: os.path.dirname(output.manifest),
    resources:
        tmpdir=lambda wildcards: f"tmp/granular/{wildcards.sample}/cluster",
    log:
        "logs/granular/{sample}/cluster.log",
    shell:
        r'''
        mkdir -p "{resources.tmpdir}" "$(dirname "{log}")"
        {params.command}           --threads "{threads}"           --tmp-dir "{resources.tmpdir}"           pipeline cluster           --outdir "{params.outdir}"           --done-marker "{output.done}"           --log-file "{log}"           --no-stdout-log
        test -s "{output.manifest}"
        test -s "{output.representative}"
        test -s "{output.done}"
        '''


rule pipeline_rescore:
    input:
        cluster=rules.pipeline_cluster.output.manifest,
        representative=rules.pipeline_cluster.output.representative,
        dock_manifests=expand("results/granular/{sample}/{engine}Files/dock_manifest.json", sample=SAMPLE, engine=ENGINES),
    output:
        results="results/granular/{sample}/rescore_results.json",
        done="results/granular/{sample}/rescore.done.json",
    threads: DEFAULT_THREADS
    conda:
        "envs/ocdocker.yml"
    params:
        command=OCDOCKER_COMMAND,
        outdir=lambda wildcards, output: os.path.dirname(output.results),
        rescoring=RESCORING_ENGINES,
    resources:
        tmpdir=lambda wildcards: f"tmp/granular/{wildcards.sample}/rescore",
    log:
        "logs/granular/{sample}/rescore.log",
    shell:
        r'''
        mkdir -p "{resources.tmpdir}" "$(dirname "{log}")"
        {params.command}           --threads "{threads}"           --tmp-dir "{resources.tmpdir}"           pipeline rescore           --outdir "{params.outdir}"           --rescoring-engines "{params.rescoring}"           --done-marker "{output.done}"           --log-file "{log}"           --no-stdout-log
        test -s "{output.results}"
        test -s "{output.done}"
        '''


rule pipeline_export:
    input:
        rescore=rules.pipeline_rescore.output.results,
    output:
        summary="results/granular/{sample}/summary.json",
        done="results/granular/{sample}/export.done.json",
    threads: 1
    conda:
        "envs/ocdocker.yml"
    params:
        command=OCDOCKER_COMMAND,
        outdir=lambda wildcards, output: os.path.dirname(output.summary),
    resources:
        tmpdir=lambda wildcards: f"tmp/granular/{wildcards.sample}/export",
    log:
        "logs/granular/{sample}/export.log",
    shell:
        r'''
        mkdir -p "{resources.tmpdir}" "$(dirname "{log}")"
        {params.command}           --threads "{threads}"           --tmp-dir "{resources.tmpdir}"           pipeline export           --outdir "{params.outdir}"           --done-marker "{output.done}"           --log-file "{log}"           --no-stdout-log
        test -s "{output.summary}"
        test -s "{output.done}"
        '''
