#!/usr/bin/env python3
"""
Example: Shared Feature Reduction for PDBbind and DUDEz Pipeline Results

CLI equivalent: ``ocdocker ocscore reduce``

Usage:
    python examples/14_feature_reduction_pdbbind_dudez.py \\
        --pdbbind-archive /path/to/pdbbind.tar.gz \\
        --dudez-archive /path/to/DUDEz.tar.gz \\
        --output-dir /path/to/feature_reduction_output
"""

from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from OCDocker.OCScore.CLI import reduce


if __name__ == "__main__":
    raise SystemExit(reduce.main())
