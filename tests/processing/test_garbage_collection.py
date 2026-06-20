#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for Processing.GarbageCollection helpers.
'''

# Imports
###############################################################################
import pytest

import OCDocker.Processing.GarbageCollection as ocgc

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
## Private ##

## Public ##

@pytest.mark.order(330)
def test_gc_collect_interval_switches_between_eager_and_batched():
    assert ocgc.gc_collect_interval(0) == 1
    assert ocgc.gc_collect_interval(ocgc.GC_COLLECT_EAGER_THRESHOLD) == 1
    assert ocgc.gc_collect_interval(ocgc.GC_COLLECT_EAGER_THRESHOLD + 1) == ocgc.GC_COLLECT_BATCH_SIZE


@pytest.mark.order(331)
def test_collect_periodically_uses_default_gc_collect(monkeypatch):
    calls = {"count": 0}
    monkeypatch.setattr(ocgc.gc, "collect", lambda: calls.__setitem__("count", calls["count"] + 1) or 0)

    ocgc.collect_periodically(processed_items=1, interval=1)
    ocgc.collect_periodically(processed_items=2, interval=3)
    ocgc.collect_periodically(processed_items=3, interval=3)

    assert calls["count"] == 2


@pytest.mark.order(332)
def test_collect_periodically_uses_custom_collector_only_on_interval():
    calls = {"count": 0}

    def _collector() -> int:
        calls["count"] += 1
        return 0

    ocgc.collect_periodically(processed_items=4, interval=2, collector=_collector)
    ocgc.collect_periodically(processed_items=5, interval=2, collector=_collector)

    assert calls["count"] == 1


@pytest.mark.order(333)
@pytest.mark.parametrize("interval", [0, -1])
def test_collect_periodically_skips_when_interval_non_positive(interval):
    calls = {"count": 0}

    def _collector() -> int:
        calls["count"] += 1
        return 0

    ocgc.collect_periodically(processed_items=10, interval=interval, collector=_collector)
    assert calls["count"] == 0


@pytest.mark.order(334)
def test_pool_chunksize_behaviour():
    assert ocgc.pool_chunksize(total_items=0, workers=4) == 1
    assert ocgc.pool_chunksize(total_items=10, workers=0) == 1
    assert ocgc.pool_chunksize(total_items=10, workers=2, factor=0) == 5
    assert ocgc.pool_chunksize(total_items=100, workers=4) >= 1
