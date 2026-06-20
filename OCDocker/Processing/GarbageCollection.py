#!/usr/bin/env python3

# Description
###############################################################################
'''
Shared garbage-collection helpers for Processing modules.
'''

# Imports
###############################################################################
import gc

from typing import Callable, Optional

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Public ##

# Constants for GC collection behavior
GC_COLLECT_BATCH_SIZE = 32     # Number of items to process before collecting GC in batch mode
GC_COLLECT_EAGER_THRESHOLD = 8 # Threshold for total items to switch from eager to batch GC collection


def gc_collect_interval(total_items: int) -> int:
    '''Choose a GC interval based on workload size.

    Small workloads keep eager collection behavior while larger batches collect
    periodically to reduce overhead.

    Parameters
    ----------
    total_items : int
        Total number of items to process, used to determine GC interval.

    Returns
    -------
    int
        Number of items to process before the next GC collection.
    '''

    if total_items <= GC_COLLECT_EAGER_THRESHOLD:
        return 1
    return GC_COLLECT_BATCH_SIZE


def collect_periodically(
    processed_items: int,
    interval: int,
    collector: Optional[Callable[[], int]] = None,
) -> None:
    '''Run ``gc.collect()`` at a fixed interval.
    
    Parameters
    ----------
    processed_items : int
        Number of items processed so far, used to determine if it's time to collect.
    interval : int
        Number of items to process before the next GC collection.
    collector : Optional[Callable[[], int]], optional
        Custom GC collection function, by default None (uses ``gc.collect``).
    '''

    if interval > 0 and processed_items % interval == 0:
        (collector or gc.collect)()


def pool_chunksize(total_items: int, workers: int, factor: int = 4) -> int:
    '''Compute a practical ``chunksize`` for ``Pool.imap_unordered``.

    Parameters
    ----------
    total_items : int
        Total number of work items.
    workers : int
        Number of pool workers.
    factor : int, optional
        Work distribution factor. Higher values produce smaller chunks.
        Default is 4.

    Returns
    -------
    int
        Chunksize value, always >= 1.
    '''

    if total_items <= 0 or workers <= 0:
        return 1
    if factor <= 0:
        factor = 1
    batch = workers * factor
    return max(1, (total_items + batch - 1) // batch)
