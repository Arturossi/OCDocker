#!/usr/bin/env python3

# Description
###############################################################################
'''
``ocdocker mcp`` command group: serve the OCDocker Workbench MCP server.
'''

# Imports
###############################################################################
from __future__ import annotations

import argparse

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################

# Duplicated from OCDocker.MCP.Server so the argparse default doesn't require
# importing the mcp/httpx/FastAPI stack just to build the CLI parser.
_DEFAULT_WORKBENCH_API_URL = "http://127.0.0.1:8765"

# Functions
###############################################################################
## Private ##


def cmd_serve(args: argparse.Namespace) -> int:
    '''Serve the OCDocker Workbench MCP server over stdio.

    Connects to an already-running ``ocdocker workbench serve`` instance;
    it does not start one. Read/plan/preview tools are always available.
    Job-execute tools (``run_job``, ``cancel_job``) require the Workbench
    job bearer token and an explicit ``confirm=True`` from the calling LLM.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    from OCDocker.MCP.Server import serve_ocdocker_mcp

    try:
        serve_ocdocker_mcp(base_url=args.workbench_api_url)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Error: could not serve OCDocker MCP: {exc}")
        return 2
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    '''Dispatch an MCP subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    handler = getattr(args, "func", None)
    if handler is None:
        return 2
    return int(handler(args))


## Public ##


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    '''Register the ``ocdocker mcp`` command group.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Main CLI subparser registry.
    '''

    parser = subparsers.add_parser(
        "mcp",
        description=(
            "Serve OCDocker over the Model Context Protocol (MCP) so LLM clients "
            "(Claude Code, Claude Desktop, ...) can inspect a Workbench workspace "
            "and design, launch, and monitor jobs.\n\n"
            "Requires a separately running `ocdocker workbench serve` instance; "
            "this command does not start one."
        ),
        help="Serve OCDocker as an MCP server for LLM clients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.set_defaults(func=cmd_mcp)
    mcp_sub = parser.add_subparsers(dest="mcp_command", required=True)

    serve = mcp_sub.add_parser(
        "serve",
        help="Serve the OCDocker MCP server over stdio",
        description=(
            "Serve the OCDocker MCP server over stdio, proxying tool calls to a "
            "running `ocdocker workbench serve` API. Read/plan/preview tools are "
            "always available; run_job and cancel_job require the Workbench job "
            "bearer token (see `ocdocker workbench serve` output) and an explicit "
            "confirm=True from the calling LLM."
        ),
    )
    serve.add_argument(
        "--workbench-api-url",
        default=_DEFAULT_WORKBENCH_API_URL,
        help=f"Base URL of a running `ocdocker workbench serve` API. Default: {_DEFAULT_WORKBENCH_API_URL}.",
    )
    serve.set_defaults(func=cmd_serve)


__all__ = [
    "cmd_mcp",
    "cmd_serve",
    "register_subparser",
]
