#!/usr/bin/env python3

# Description
###############################################################################
"""
MCP server package exposing the OCDocker Workbench API to LLM clients.
"""

# Imports
###############################################################################
from OCDocker.MCP.Server import DEFAULT_WORKBENCH_API_URL
from OCDocker.MCP.Server import MCP_SERVER_NAME
from OCDocker.MCP.Server import OCDockerMCPError
from OCDocker.MCP.Server import build_ocdocker_mcp_server
from OCDocker.MCP.Server import serve_ocdocker_mcp

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

__all__ = [
    "DEFAULT_WORKBENCH_API_URL",
    "MCP_SERVER_NAME",
    "OCDockerMCPError",
    "build_ocdocker_mcp_server",
    "serve_ocdocker_mcp",
]
