#!/usr/bin/env python3

# Description
###############################################################################
'''
Public API helpers for scheduler-friendly OCDocker workflows.
'''

# Imports
###############################################################################
from OCDocker.API.Pipeline import PipelineRunResult, run_pipeline

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

__all__ = ["PipelineRunResult", "run_pipeline"]
