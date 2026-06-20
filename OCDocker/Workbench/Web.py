#!/usr/bin/env python3

# Description
###############################################################################
"""
Packaged browser assets for the strict OCScore Workbench dashboard.
"""

# Imports
###############################################################################
from __future__ import annotations

from pathlib import Path
from typing import Final

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Constants
###############################################################################

WORKBENCH_WEB_INDEX_ROUTE: Final[str] = "/app"
WORKBENCH_WEB_FAVICON_ROUTE: Final[str] = "/app-favicon.png"
WORKBENCH_WEB_BRAND_LOGO_ROUTE: Final[str] = "/app-brand-logo.png"
WORKBENCH_WEB_ROUTES: Final[tuple[str, ...]] = (
    "/app",
    "/app/",
    "/app.css",
    "/app.js",
    WORKBENCH_WEB_FAVICON_ROUTE,
    WORKBENCH_WEB_BRAND_LOGO_ROUTE,
)
WORKBENCH_STATIC_DIR: Final[Path] = Path(__file__).resolve().parent / "static"

_WEB_ASSET_FILES: Final[dict[str, tuple[str, str]]] = {
    "/app": ("text/html; charset=utf-8", "index.html"),
    "/app.css": ("text/css; charset=utf-8", "app.css"),
    "/app.js": ("text/javascript; charset=utf-8", "app.js"),
}

# Functions
###############################################################################
## Private ##


def _workbench_static_path(filename: str) -> Path:
    '''Return one packaged Workbench static asset path.

    Parameters
    ----------
    filename : str
        Static asset filename relative to ``WORKBENCH_STATIC_DIR``.

    Returns
    -------
    pathlib.Path
        Resolved static asset path.
    '''

    return WORKBENCH_STATIC_DIR / filename


def _read_workbench_static_file(filename: str) -> bytes:
    '''Read one Workbench static asset from disk.

    Parameters
    ----------
    filename : str
        Static asset filename relative to ``WORKBENCH_STATIC_DIR``.

    Returns
    -------
    bytes
        Raw asset bytes.

    Raises
    ------
    FileNotFoundError
        If the packaged asset is missing from the installed package tree.
    '''

    path = _workbench_static_path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"Workbench static asset not found: {path}")
    return path.read_bytes()


def _workbench_repo_root() -> Path:
    '''Return the repository root beside the installed ``OCDocker`` package.'''

    return Path(__file__).resolve().parents[2]


def _workbench_favicon_path() -> Path | None:
    '''Return the browser tab icon path when available beside the package root.

    Returns
    -------
    pathlib.Path or None
        Favicon path when ``ocdocker_small_logo.png`` exists beside the repo root.
    '''

    candidate = _workbench_repo_root() / "ocdocker_small_logo.png"
    return candidate if candidate.is_file() else None


def _workbench_brand_logo_path() -> Path | None:
    '''Return the in-page OCDocker wordmark path when available beside the package root.

    Returns
    -------
    pathlib.Path or None
        Brand logo path when ``OCDocker.png`` exists beside the repo root or
        under ``OCDocker/Workbench/assets/``.
    '''

    candidates = (
        _workbench_repo_root() / "OCDocker.png",
        Path(__file__).resolve().parent / "assets" / "OCDocker.png",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


## Public ##


def is_workbench_web_asset_path(path: str) -> bool:
    '''Return whether a path is served by packaged Workbench web assets.

    Parameters
    ----------
    path : str
        Request path.

    Returns
    -------
    bool
        True when the path is a packaged web asset route.
    '''

    return path in WORKBENCH_WEB_ROUTES


def build_workbench_web_asset(path: str) -> tuple[str, bytes]:
    '''Load one packaged Workbench browser asset.

    Parameters
    ----------
    path : str
        Request path.

    Returns
    -------
    tuple[str, bytes]
        Content type and response body.

    Raises
    ------
    KeyError
        If the request path is not a known Workbench web asset route.
    FileNotFoundError
        If a packaged static asset is missing from the installed package tree.
    '''

    route = WORKBENCH_WEB_INDEX_ROUTE if path == "/app/" else path
    static_asset = _WEB_ASSET_FILES.get(route)
    if static_asset is not None:
        content_type, filename = static_asset
        return content_type, _read_workbench_static_file(filename)
    if route == WORKBENCH_WEB_FAVICON_ROUTE:
        favicon_path = _workbench_favicon_path()
        if favicon_path is None:
            raise KeyError(f"Unknown Workbench web asset: {path}")
        return "image/png", favicon_path.read_bytes()
    if route == WORKBENCH_WEB_BRAND_LOGO_ROUTE:
        brand_logo_path = _workbench_brand_logo_path()
        if brand_logo_path is None:
            raise KeyError(f"Unknown Workbench web asset: {path}")
        return "image/png", brand_logo_path.read_bytes()
    raise KeyError(f"Unknown Workbench web asset: {path}")


__all__ = [
    "WORKBENCH_STATIC_DIR",
    "WORKBENCH_WEB_INDEX_ROUTE",
    "WORKBENCH_WEB_FAVICON_ROUTE",
    "WORKBENCH_WEB_BRAND_LOGO_ROUTE",
    "WORKBENCH_WEB_ROUTES",
    "build_workbench_web_asset",
    "is_workbench_web_asset_path",
]
