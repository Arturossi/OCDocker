#!/usr/bin/env python3

# Description
###############################################################################
'''Leakage audit helpers for staged OCScore production-grade protocols.'''

# Imports
###############################################################################
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import OCDocker.Toolbox.Logging as oclogging

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

LOGGER = oclogging.get_logger("ocscore.utils.leakage_audit")


@dataclass
class LeakageFinding:
    """Single leakage audit finding."""

    severity: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class LeakageAuditResult:
    """Aggregated leakage audit output."""

    passed: bool
    findings: list[LeakageFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        '''Serialize leakage audit findings for JSON reports.

        Returns
        -------
        dict[str, Any]
            Pass/fail flag and structured finding list.
        '''

        return {
            "passed": self.passed,
            "findings": [
                {"severity": f.severity, "code": f.code, "message": f.message, "details": f.details}
                for f in self.findings
            ],
        }


def _receptor_overlap(receptors: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, int]:
    train_r = set(receptors[train_idx].tolist())
    val_r = set(receptors[val_idx].tolist())
    test_r = set(receptors[test_idx].tolist())
    return {
        "train∩validation": len(train_r & val_r),
        "train∩test": len(train_r & test_r),
        "validation∩test": len(val_r & test_r),
    }


def run_leakage_audit(
        *,
        feature_selection: Optional[dict[str, Any]] = None,
        pdbbind_split_diagnostics: Optional[dict[str, Any]] = None,
        dudez_split_diagnostics: Optional[dict[str, Any]] = None,
        scaling_metadata: Optional[dict[str, Any]] = None,
        strict: bool = False,
    ) -> LeakageAuditResult:
    '''Run production-grade leakage checks on staged protocol metadata.

    Parameters
    ----------
    feature_selection : dict | None
        Feature-selection scope metadata.
    pdbbind_split_diagnostics : dict | None
        PDBbind split diagnostics from the regression stage.
    dudez_split_diagnostics : dict | None
        DUDEz split diagnostics from the screening stage.
    scaling_metadata : dict | None
        DUDEz scaling metadata.
    strict : bool
        When True, critical findings fail the audit.

    Returns
    -------
    LeakageAuditResult
        Audit outcome and findings list.
    '''

    findings: list[LeakageFinding] = []

    if feature_selection is not None and feature_selection.get("scope") == "precomputed_global":
        findings.append(LeakageFinding(
            severity="error",
            code="feature_selection_precomputed_global",
            message="Features were selected on merged data before modeling splits.",
            details={"scope": feature_selection.get("scope"), "fit_dataset": feature_selection.get("fit_dataset")},
        ))

    for label, diagnostics in (("pdbbind", pdbbind_split_diagnostics), ("dudez", dudez_split_diagnostics)):
        if not diagnostics:
            continue
        overlap = diagnostics.get("receptor_overlap") or {}
        if any(int(value) > 0 for value in overlap.values()):
            strategy = str(diagnostics.get("strategy", ""))
            severity = "error" if "heldout" in strategy else "warning"
            findings.append(LeakageFinding(
                severity=severity,
                code=f"{label}_receptor_overlap",
                message=f"{label} split has receptor overlap across partitions.",
                details={"strategy": diagnostics.get("strategy"), "overlap": overlap},
            ))

    critical = [finding for finding in findings if finding.severity == "error"]
    passed = len(critical) == 0
    if strict and not passed:
        LOGGER.error("Leakage audit failed with %s critical findings.", len(critical))
    return LeakageAuditResult(passed=passed, findings=findings)


def write_leakage_audit_report(output_dir: str | Path, result: LeakageAuditResult) -> Path:
    '''Write ``leakage_audit.json`` to ``output_dir``.'''

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "leakage_audit.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "LeakageAuditResult",
    "LeakageFinding",
    "run_leakage_audit",
    "write_leakage_audit_report",
]
