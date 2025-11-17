import os

import pytest

import OCDocker.Toolbox.IO as ocio


@pytest.mark.order(68)
def test_lazyread_and_reverse(tmp_path):
    p = tmp_path / "sample.txt"
    lines = [f"line-{i}" for i in range(5)]
    p.write_text("\n".join(lines) + "\n")

    # Forward mmap reader
    fwd = list(ocio.lazyread_mmap(str(p)))
    assert [s.strip() for s in fwd] == lines

    # Reverse mmap reader
    rev = list(ocio.lazyread_reverse_order_mmap(str(p)))
    assert [s.strip() for s in rev] == list(reversed(lines))

    # Non-mmap variants
    fwd2 = list(ocio.lazyread(str(p)))
    assert [s.strip() for s in fwd2] == lines
    rev2 = list(ocio.lazyread_reverse_order(str(p)))
    assert [s.strip() for s in rev2] == list(reversed(lines))

