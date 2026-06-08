"""OCScore staged-pipeline CLI subcommands."""

from __future__ import annotations

import argparse

from OCDocker.OCScore.CLI import export_tools
from OCDocker.OCScore.CLI import reduce
from OCDocker.OCScore.CLI import train


def register_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register reduce, train, and export-tool subcommands."""

    reduce.register_subparser(subparsers)
    train.register_subparser(subparsers)
    export_tools.register_subparsers(subparsers)
