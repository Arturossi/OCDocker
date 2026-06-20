#!/usr/bin/env python3

# Description
###############################################################################
'''
Cryptographic content hashes for OCScore provenance artifacts.

Usage:

from OCDocker.OCScore.Utils.ContentHash import hash_feature_list
from OCDocker.OCScore.Utils.ContentHash import hash_file
'''

from __future__ import annotations

# Imports
###############################################################################
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Functions
###############################################################################
## Public ##


def hash_bytes(payload: bytes) -> str:
    '''Return the SHA-256 hex digest of ``payload``.'''

    return hashlib.sha256(payload).hexdigest()


def hash_text(text: str) -> str:
    '''Return the SHA-256 hex digest of UTF-8 encoded text.'''

    return hash_bytes(text.encode("utf-8"))


def hash_feature_list(features: Sequence[str]) -> str:
    '''Return a stable hash for an ordered feature-name list.'''

    payload = json.dumps(list(features), separators=(",", ":"), ensure_ascii=True)
    return hash_text(payload)


def hash_json_dict(payload: dict[str, Any]) -> str:
    '''Return a stable hash for a JSON-serializable mapping.'''

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hash_text(encoded)


def hash_split_indices(indices: Sequence[int]) -> str:
    '''Return a stable hash for an ordered index list.'''

    payload = json.dumps([int(value) for value in indices], separators=(",", ":"), ensure_ascii=True)
    return hash_text(payload)


def hash_dataframe_partition(
        df: pd.DataFrame,
        indices: Sequence[int],
        *,
        id_columns: Sequence[str] = ("name", "receptor", "ligand"),
    ) -> str:
    '''Return a stable hash for dataframe rows referenced by ``indices``.'''

    frame = df.iloc[list(indices)].copy()
    available = [column for column in id_columns if column in frame.columns]
    if available:
        rows = frame[available].astype(str).fillna("").to_dict(orient="records")
    else:
        rows = [{"row_index": int(index)} for index in indices]
    return hash_json_dict({"rows": rows})


def hash_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    '''Return the SHA-256 hex digest of a file's contents.'''

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "hash_bytes",
    "hash_dataframe_partition",
    "hash_feature_list",
    "hash_file",
    "hash_json_dict",
    "hash_split_indices",
    "hash_text",
]
