#!/usr/bin/env python3

# Description
###############################################################################
"""
SQLAlchemy model for pipeline post-processing metadata.

Usage:

from OCDocker.DB.Models.PipelineRuns import PipelineRuns
"""

# Imports
###############################################################################
from sqlalchemy import Column, Integer, String, Text

from OCDocker.DB.Models.Base import base

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Classes
###############################################################################


class PipelineRuns(base):
    """Define the table for rich pipeline run metadata.

    Attributes
    ----------
    complex_id : Integer
        Optional foreign key reference to the Complexes row ID.
    representative_pose : String
        Path to the selected representative pose for this run.
    representative_engine : String
        Engine label associated with the selected representative pose.
    rescoring_json : Text
        JSON payload with full rescoring outputs for all executed engines.
    summary_json : Text
        JSON summary produced by pipeline post-processing.
    payload_path : String
        Path to payload.pkl for this run.
    run_report_path : String
        Path to run_report.json for this run.
    """

    complex_id = Column(Integer, nullable=True, index=True)
    representative_pose = Column(String(2048), nullable=True)
    representative_engine = Column(String(64), nullable=True)
    rescoring_json = Column(Text, nullable=True)
    summary_json = Column(Text, nullable=True)
    payload_path = Column(String(2048), nullable=True)
    run_report_path = Column(String(2048), nullable=True)
