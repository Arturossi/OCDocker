#!/usr/bin/env python3
"""init-config CLI command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

def cmd_init_config(args: argparse.Namespace) -> int:
    '''Create a base OCDocker config file from an example template.

    This avoids importing Initialise (which expects a ready config).

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    '''

    target = Path(args.config_file or "OCDocker.cfg")
    suffix = target.suffix.lower()
    if suffix == ".yml":
        example_names = ("OCDocker.yml.example", "OCDocker.cfg.example")
    else:
        # Default to default cfg template when extension is absent/unknown.
        example_names = ("OCDocker.cfg.example", "OCDocker.yml.example")

    def _find_example(base_dir: Path) -> Optional[Path]:
        for name in example_names:
            candidate = base_dir / name
            if candidate.exists():
                return candidate
        return None

    # Look for the example file in current directory first.
    example = _find_example(Path("."))
    if example is None:
        # Fallback to package directory.
        import OCDocker
        pkg_dir = Path(OCDocker.__file__).parent.parent
        example = _find_example(pkg_dir)
        if example is None:
            expected = " or ".join(example_names)
            print(f"{expected} not found in current directory or package directory.")
            return 1

    if target.exists():
        print(f"Config already exists: {target}")
        return 0

    target.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Config created at: {target}. Please review and adjust paths.")
    return 0


def register_subparser(sub: argparse._SubParsersAction, parent: argparse.ArgumentParser) -> None:
    p_init = sub.add_parser(
        "init-config",
        description=(
            "Create a starter OCDocker.cfg or OCDocker.yml configuration file from the example template.\n"
            "This command copies the matching example template into the target config file,\n"
            "allowing you to customize paths to docking binaries, databases, and other settings."
        ),
        help="Create a starter OCDocker config file (.cfg or .yml)",
        parents=[parent],
    )
    p_init.set_defaults(func=cmd_init_config)
