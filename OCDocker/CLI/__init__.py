#!/usr/bin/env python3

# Description
###############################################################################
'''
Unified command-line interface for OCDocker tasks.

Main commands
-------------
- version: prints library version.
- manifest: emits a reproducibility manifest with runtime/tool/package versions.
- init-config: creates a quick `OCDocker.cfg` or `OCDocker.yml` from the example file.
- vs: runs docking and optional rescoring for one receptor/ligand/box using Vina, Smina, or PLANTS.
- ocscore: staged OCScore ML pipeline (reduce, train, export tools, shap).
- workbench: validates Workbench specs and emits command plans without execution.
- pipeline: full multi-engine flow — run docking across engines, cluster poses by RMSD,
            pick the representative pose (medoid of the largest cluster), rescore and export results.

Global options
--------------
- --conf, --multiprocess, --update-databases, --output-level, --overwrite, --no-splash:
  compatible with OCDocker.Initialise and used to bootstrap the environment.

Modules
-------
- __init__: CLI entry points, command parsing, and dispatch.
- parser: Argument parser construction.
- common, workflow: Shared helpers for docking commands.
- vs, pipeline, doctor, manifest, console, script, init_config, workbench: Command implementations.
- ocscore: OCScore command group registration.
'''

from __future__ import annotations

import sys
from typing import Optional

from OCDocker.CLI.parser import build_parser

__all__ = [
    "main",
    "build_parser",
]


def main(argv: Optional[list[str]] = None) -> int:
    '''Main entry point for the CLI.'''

    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
