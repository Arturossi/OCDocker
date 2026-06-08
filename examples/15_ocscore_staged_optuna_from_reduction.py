#!/usr/bin/env python3
"""
Example: Staged OCScore Optuna from raw unreduced inputs.

CLI equivalent: ``ocdocker ocscore train``

Usage:
    python examples/15_ocscore_staged_optuna_from_reduction.py \
        --protocol development \
        --raw-input-dir /path/to/raw_prepare \
        --output-dir /path/to/optuna_output
"""

from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from OCDocker.OCScore.CLI import train


if __name__ == "__main__":
    raise SystemExit(train.main())
