#!/usr/bin/env python3

# Description
###############################################################################
'''
Publication-oriented OCScore architecture diagrams.

Usage:

import OCDocker.OCScore.Analysis.Plotting.ArchitecturePlots as ocarchplot
'''

# Imports
###############################################################################
from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import yaml

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import OCDocker.OCScore.Optimization.ModelExport as ocexport


# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################

@dataclass(frozen=True)
class ArchitectureBlock:
    """Single visible block in an architecture diagram.

    Attributes
    ----------
    label : str
        Text label shown in the block.
    dim : int or None
        Feature width or output dimensionality represented by the block.
    kind : str
        Visual role used for colors, such as ``input``, ``encoder``, or
        ``head``.
    detail : str, default=""
        Optional compact annotation shown below the dimensionality.
    """

    label: str
    dim: int | None
    kind: str
    detail: str = ""


@dataclass(frozen=True)
class ArchitectureDiagram:
    """Normalized architecture diagram specification.

    Attributes
    ----------
    task : str
        Name of the task or model family represented by the diagram.
    main : tuple of ArchitectureBlock
        Primary left-to-right model path.
    auxiliary : tuple of ArchitectureBlock, default=()
        Optional auxiliary branch, used for reconstruction decoders.
    notes : tuple of str, default=()
        Compact notes rendered below the diagram.
    """

    task: str
    main: tuple[ArchitectureBlock, ...]
    auxiliary: tuple[ArchitectureBlock, ...] = ()
    notes: tuple[str, ...] = ()


# Functions
###############################################################################
## Private ##

_COLORS = {
    "input": ("#f3f4f6", "#374151"),
    "encoder": ("#dbeafe", "#1d4ed8"),
    "latent": ("#ccfbf1", "#0f766e"),
    "projection": ("#fef3c7", "#b45309"),
    "head": ("#dcfce7", "#15803d"),
    "decoder": ("#e0e7ff", "#4338ca"),
    "output": ("#fce7f3", "#be185d"),
}


def _as_int(value: Any) -> int | None:
    '''Convert a value to an integer when possible.

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    int or None
        Integer value, or None when the input is missing or invalid.
    '''

    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_dim(dim: int | None) -> str:
    '''Format a layer dimension for display.

    Parameters
    ----------
    dim : int or None
        Layer dimensionality.

    Returns
    -------
    str
        Human-readable dimensionality label.
    '''

    return "n/a" if dim is None else str(int(dim))


def _format_task_title(task: str) -> str:
    '''Format an architecture task name for figure titles.

    Parameters
    ----------
    task : str
        Raw task name from the architecture document.

    Returns
    -------
    str
        Publication-oriented task title.
    '''

    titles = {
        "dudez_screening": "DUDEz Classification",
        "pdbbind_regression": "PDBbind Regression",
    }
    return titles.get(str(task), str(task).replace("_", " ").title())


def _read_architecture_file(path: Path) -> dict[str, Any]:
    '''Read a JSON or YAML architecture file.

    Parameters
    ----------
    path : pathlib.Path
        Architecture file path.

    Returns
    -------
    dict
        Parsed architecture document.

    Raises
    ------
    ValueError
        If the file content is not a mapping.
    '''

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yml", ".yaml"}:
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Architecture file must contain a mapping: {path}")
    return loaded


def _hidden_blocks(
    prefix: str, sizes: Iterable[Any], *, kind: str, detail: str = ""
) -> list[ArchitectureBlock]:
    '''Build repeated hidden-layer diagram blocks.

    Parameters
    ----------
    prefix : str
        Prefix used to label each block.
    sizes : Iterable
        Hidden-layer dimensionalities.
    kind : str
        Visual block kind.
    detail : str, default=""
        Optional annotation copied to each block.

    Returns
    -------
    list of ArchitectureBlock
        Diagram blocks for the hidden layers.
    '''

    blocks = []
    for idx, size in enumerate(sizes, start=1):
        blocks.append(ArchitectureBlock(f"{prefix} {idx}", _as_int(size), kind, detail))
    return blocks


def _layer_neuron_count(dim: int | None, *, kind: str) -> int:
    '''Choose the number of visible neurons for a layer.

    Parameters
    ----------
    dim : int or None
        Real layer dimensionality.
    kind : str
        Layer kind used to keep output layers compact.

    Returns
    -------
    int
        Number of representative neurons drawn in the figure.
    '''

    if kind == "output" or dim == 1:
        return 1
    if dim is None:
        return 5
    if dim <= 8:
        return max(2, int(dim))
    return 7


def _layer_color(kind: str) -> tuple[str, str, str]:
    '''Return face, edge, and connection colors for a layer kind.

    Parameters
    ----------
    kind : str
        Layer kind.

    Returns
    -------
    tuple of str
        Face color, edge color, and connector color.
    '''

    face, edge = _COLORS.get(kind, _COLORS["encoder"])
    connector = {
        "input": "#6b7280",
        "encoder": "#2563eb",
        "latent": "#0f766e",
        "projection": "#b45309",
        "head": "#15803d",
        "decoder": "#4f46e5",
        "output": "#be185d",
    }.get(kind, "#4b5563")
    return face, edge, connector


def _neuron_positions(
    x: float, y: float, count: int, spacing: float = 0.22
) -> list[tuple[float, float]]:
    '''Build coordinates for representative neurons.

    Parameters
    ----------
    x, y : float
        Center of the visible layer column.
    count : int
        Number of visible neurons.
    spacing : float, default=0.22
        Vertical spacing between neurons.

    Returns
    -------
    list of tuple of float
        Neuron center coordinates.
    '''

    offset = (count - 1) * spacing / 2.0
    return [(x, y + offset - idx * spacing) for idx in range(count)]


def _draw_layer_column(
    ax: plt.Axes,
    block: ArchitectureBlock,
    x: float,
    y: float,
    *,
    radius: float = 0.065,
) -> list[tuple[float, float]]:
    '''Draw a neural-network layer as representative neurons.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes receiving the layer.
    block : ArchitectureBlock
        Layer metadata.
    x, y : float
        Layer center coordinates.
    radius : float, default=0.065
        Neuron circle radius.

    Returns
    -------
    list of tuple of float
        Visible neuron positions for connection drawing.
    '''

    face, edge, _ = _layer_color(block.kind)
    count = _layer_neuron_count(block.dim, kind=block.kind)
    positions = _neuron_positions(x, y, count)
    if positions:
        xs, ys = zip(*positions)
        ax.scatter(
            xs,
            ys,
            s=68,
            marker="o",
            facecolors=face,
            edgecolors=edge,
            linewidths=1.25,
            alpha=0.98,
            zorder=3,
        )
    if block.dim is not None and block.dim > count:
        mid_idx = len(positions) // 2
        mid_x, mid_y = positions[mid_idx]
        ax.scatter(
            [mid_x],
            [mid_y],
            s=78,
            marker="o",
            facecolors="#ffffff",
            edgecolors=edge,
            linewidths=1.15,
            zorder=4,
        )
        ax.text(
            mid_x,
            mid_y + radius * 0.03,
            "...",
            ha="center",
            va="center",
            fontsize=5.6,
            color=edge,
            weight="bold",
            zorder=5,
        )
    ax.text(
        x,
        y + 1.18,
        block.label,
        ha="center",
        va="center",
        fontsize=8.8,
        color="#111827",
        weight="bold",
    )
    ax.text(
        x,
        y - 1.12,
        f"dim={_format_dim(block.dim)}",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#374151",
    )
    if block.detail:
        ax.text(
            x,
            y - 1.32,
            block.detail,
            ha="center",
            va="center",
            fontsize=7.4,
            color="#6b7280",
        )
    return positions


def _draw_dense_connections(
    ax: plt.Axes,
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
    *,
    color: str = "#64748b",
    alpha: float = 0.16,
    direction_y: float | None = None,
) -> None:
    '''Draw representative dense connections between two layers.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes receiving the connections.
    left, right : sequence of tuple of float
        Source and target neuron coordinates.
    color : str, default="#64748b"
        Line color.
    alpha : float, default=0.16
        Connection transparency.
    direction_y : float, optional
        Vertical position for the directional arrow. When omitted, the arrow is
        drawn at the connection midpoint.
    '''

    if not left or not right:
        return
    for lx, ly in left:
        for rx, ry in right:
            ax.plot(
                [lx, rx],
                [ly, ry],
                color=color,
                linewidth=0.42,
                alpha=alpha,
                zorder=1,
            )
    left_mid = sum(y for _, y in left) / len(left)
    right_mid = sum(y for _, y in right) / len(right)
    arrow_y = direction_y if direction_y is not None else (left_mid + right_mid) / 2.0
    arrow = FancyArrowPatch(
        (left[0][0] + 0.18, arrow_y),
        (right[0][0] - 0.18, arrow_y),
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=0.9,
        color=color,
        alpha=0.62,
        zorder=2,
    )
    ax.add_patch(arrow)


def _draw_group_band(
    ax: plt.Axes,
    label: str,
    x_min: float,
    x_max: float,
    y: float,
    height: float,
    color: str,
) -> None:
    '''Draw a subtle background band for a model section.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes receiving the band.
    label : str
        Band label.
    x_min, x_max : float
        Horizontal extent.
    y : float
        Band center coordinate.
    height : float
        Band height.
    color : str
        Band edge and label color.
    '''

    patch = FancyBboxPatch(
        (x_min, y - height / 2.0),
        x_max - x_min,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=0.85,
        edgecolor=color,
        facecolor=color,
        alpha=0.055,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        x_min + 0.08,
        y + height / 2.0 + 0.12,
        label,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color=color,
        weight="bold",
        alpha=0.84,
    )


def _draw_branch_arrow(
    ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    '''Draw a curved auxiliary branch arrow.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes receiving the arrow.
    start : tuple of float
        Arrow start coordinates.
    end : tuple of float
        Arrow end coordinates.
    '''

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=-0.22",
        mutation_scale=10,
        linewidth=1.15,
        color="#4f46e5",
        alpha=0.76,
        zorder=2,
    )
    ax.add_patch(arrow)


## Public ##

def normalize_architecture(document: Mapping[str, Any]) -> ArchitectureDiagram:
    '''Normalize architecture metadata for plotting.

    Parameters
    ----------
    document : Mapping
        Exported OCScore architecture document or a manual architecture mapping.

    Returns
    -------
    ArchitectureDiagram
        Diagram-ready architecture with primary and optional auxiliary blocks.

    Raises
    ------
    ValueError
        If a manual ``layers`` section is present but is not a list.
    '''

    task = str(document.get("task") or document.get("name") or "architecture")
    input_dim = _as_int(document.get("input_size") or document.get("input_dim"))
    notes: list[str] = []

    if "layers" in document:
        raw_layers = document["layers"]
        if not isinstance(raw_layers, Sequence):
            raise ValueError("architecture.layers must be a list.")
        blocks = []
        for idx, layer in enumerate(raw_layers, start=1):
            if isinstance(layer, Mapping):
                label = str(layer.get("label") or layer.get("name") or f"Layer {idx}")
                dim = _as_int(
                    layer.get("dim") or layer.get("size") or layer.get("units")
                )
                kind = str(layer.get("kind") or "encoder")
                detail = str(layer.get("detail") or layer.get("activation") or "")
            else:
                label = f"Layer {idx}"
                dim = _as_int(layer)
                kind = "encoder"
                detail = ""
            blocks.append(ArchitectureBlock(label, dim, kind, detail))
        return ArchitectureDiagram(task=task, main=tuple(blocks), notes=tuple(notes))

    if task == "dudez_screening" or "feature_extractor" in document:
        extractor = dict(document.get("feature_extractor") or {})
        classifier = dict(document.get("classifier") or {})
        transfer = dict(document.get("transfer") or {})
        main = [ArchitectureBlock("Input", input_dim, "input")]
        main.extend(
            _hidden_blocks("Feature", extractor.get("hidden_sizes", []), kind="encoder")
        )
        latent_dim = _as_int(extractor.get("latent_dim"))
        if latent_dim is not None:
            main.append(ArchitectureBlock("Latent", latent_dim, "latent"))
        projection_dim = _as_int(extractor.get("projection_dim"))
        output_dim = _as_int(extractor.get("output_dim"))
        if projection_dim:
            main.append(ArchitectureBlock("Projection", projection_dim, "projection"))
        elif output_dim is not None and output_dim != latent_dim:
            main.append(ArchitectureBlock("Embedding", output_dim, "projection"))
        hidden = _as_int(classifier.get("hidden_size"))
        if hidden:
            detail = str(classifier.get("activation") or "")
            main.append(ArchitectureBlock("Classifier", hidden, "head", detail))
        main.append(ArchitectureBlock("Probability", 1, "output"))
        if transfer:
            notes.append(
                "Transfer: "
                f"{'enabled' if transfer.get('use_transfer', False) else 'disabled'}, "
                f"mode={transfer.get('fine_tuning_mode', 'n/a')}"
            )
        return ArchitectureDiagram(task=task, main=tuple(main), notes=tuple(notes))

    encoder = dict(document.get("encoder") or {})
    projection = dict(document.get("projection") or {})
    decoder = dict(document.get("decoder") or {})
    dae = dict(document.get("dae") or {})
    head = dict(document.get("regression_head") or {})
    resolved = dict(encoder.get("resolved") or {})
    hidden_sizes = resolved.get("hidden_sizes", encoder.get("hidden_sizes", []))
    latent_dim = _as_int(resolved.get("latent_dim", encoder.get("latent_dim")))
    projection_dim = _as_int(
        projection.get("projection_dim") or resolved.get("projection_dim")
    )

    main = [ArchitectureBlock("Input", input_dim, "input")]
    detail = str(encoder.get("activation") or "")
    main.extend(_hidden_blocks("Encoder", hidden_sizes, kind="encoder", detail=detail))
    if latent_dim is not None:
        main.append(ArchitectureBlock("Latent", latent_dim, "latent"))
    if projection.get("enabled", bool(projection_dim)) and projection_dim:
        main.append(ArchitectureBlock("Projection", projection_dim, "projection"))
    main.append(ArchitectureBlock("Affinity", 1, "head", str(head.get("loss") or "")))

    aux: list[ArchitectureBlock] = []
    if decoder.get("enabled", bool(decoder.get("hidden_sizes"))):
        aux.extend(
            _hidden_blocks("Decoder", decoder.get("hidden_sizes", []), kind="decoder")
        )
        aux.append(ArchitectureBlock("Reconstruction", input_dim, "output"))
        notes.append(f"Decoder lambda={decoder.get('lambda_rec', 'n/a')}")
    if dae.get("enabled"):
        notes.append(
            f"DAE: {dae.get('noise_type', 'none')}, "
            f"mask={dae.get('mask_prob', 0)}, std={dae.get('gaussian_std', 0)}"
        )
    return ArchitectureDiagram(
        task=task, main=tuple(main), auxiliary=tuple(aux), notes=tuple(notes)
    )


def load_architecture_document(source: str | Path) -> tuple[dict[str, Any], Path]:
    '''Load an architecture document.

    Parameters
    ----------
    source : str or pathlib.Path
        Architecture JSON/YAML file or a ``best_model/`` export directory.

    Returns
    -------
    tuple of dict and pathlib.Path
        Parsed architecture document and the resolved architecture file path.

    Raises
    ------
    FileNotFoundError
        If the source path or export architecture file does not exist.
    ValueError
        If the architecture file does not contain a mapping.
    '''

    path = Path(source)
    if path.is_dir():
        arch_path = path / ocexport.ARCHITECTURE_FILENAME
        if not arch_path.exists():
            raise FileNotFoundError(
                f"Missing {ocexport.ARCHITECTURE_FILENAME} in {path}"
            )
        return _read_architecture_file(arch_path), arch_path
    if not path.exists():
        raise FileNotFoundError(str(path))
    return _read_architecture_file(path), path


def plot_architecture_diagram(
    architecture: Mapping[str, Any] | ArchitectureDiagram,
    *,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    include_decoder: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    '''Build a publication-oriented neural-network architecture figure.

    Parameters
    ----------
    architecture : Mapping or ArchitectureDiagram
        Architecture metadata or a pre-normalized diagram specification.
    title : str, optional
        Custom title. When omitted, the task name is used.
    figsize : tuple of float, optional
        Explicit matplotlib figure size.
    include_decoder : bool, default=False
        Whether to include the auxiliary reconstruction decoder branch.

    Returns
    -------
    tuple of matplotlib.figure.Figure and matplotlib.axes.Axes
        Figure and axes containing the rendered architecture diagram.

    Raises
    ------
    ValueError
        If the architecture has no layers to plot.
    '''

    diagram = (
        architecture
        if isinstance(architecture, ArchitectureDiagram)
        else normalize_architecture(architecture)
    )
    blocks = list(diagram.main)
    if not blocks:
        raise ValueError("Architecture has no layers to plot.")

    auxiliary_blocks = list(diagram.auxiliary) if include_decoder else []
    n_main = len(blocks)
    x_gap = 1.18 if n_main > 8 else 1.34
    x_positions = [idx * x_gap for idx in range(n_main)]
    y_main = 0.0
    n_aux = len(auxiliary_blocks)
    figure_width = max(8.8, x_gap * (n_main + max(0, n_aux - 1)) + 2.8)
    figure_height = 6.2 if auxiliary_blocks else 4.3
    if figsize is not None:
        figure_width, figure_height = figsize

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")

    encoder_indices = [
        idx
        for idx, block in enumerate(blocks)
        if block.kind in {"encoder", "latent", "projection"}
    ]
    head_indices = [
        idx for idx, block in enumerate(blocks) if block.kind in {"head", "output"}
    ]
    if encoder_indices:
        _draw_group_band(
            ax,
            "representation learning",
            x_positions[min(encoder_indices)] - 0.36,
            x_positions[max(encoder_indices)] + 0.36,
            y_main,
            2.72,
            "#2563eb",
        )
    if head_indices:
        _draw_group_band(
            ax,
            "prediction head",
            x_positions[min(head_indices)] - 0.36,
            x_positions[max(head_indices)] + 0.36,
            y_main,
            2.72,
            "#15803d",
        )

    direction_y = y_main - 0.78
    layer_positions: list[list[tuple[float, float]]] = []
    for idx, block in enumerate(blocks):
        positions = _draw_layer_column(ax, block, x_positions[idx], y_main)
        layer_positions.append(positions)
        if idx:
            _, _, connector = _layer_color(block.kind)
            _draw_dense_connections(
                ax,
                layer_positions[idx - 1],
                layer_positions[idx],
                color=connector,
                direction_y=direction_y,
                alpha=(
                    0.11
                    if len(layer_positions[idx - 1]) * len(layer_positions[idx]) > 35
                    else 0.18
                ),
            )

    y_min = -1.8
    all_x_positions = list(x_positions)
    if auxiliary_blocks:
        y_aux = -3.05
        branch_candidates = [
            idx
            for idx, block in enumerate(blocks)
            if block.kind in {"latent", "projection"}
        ]
        start_idx = branch_candidates[-1] if branch_candidates else max(0, n_main - 2)
        aux_positions = [
            x_positions[start_idx] + (idx + 1) * max(x_gap, 1.55)
            for idx in range(len(auxiliary_blocks))
        ]
        _draw_group_band(
            ax,
            "auxiliary reconstruction",
            aux_positions[0] - 0.36,
            aux_positions[-1] + 0.36,
            y_aux,
            2.55,
            "#4f46e5",
        )
        _draw_branch_arrow(
            ax,
            (x_positions[start_idx], y_main - 0.92),
            (aux_positions[0] - 0.18, y_aux + 0.88),
        )
        aux_direction_y = y_aux - 0.78
        previous = layer_positions[start_idx]
        for idx, block in enumerate(auxiliary_blocks):
            positions = _draw_layer_column(ax, block, aux_positions[idx], y_aux)
            if idx:
                _draw_dense_connections(
                    ax,
                    previous,
                    positions,
                    color="#4f46e5",
                    alpha=0.13,
                    direction_y=aux_direction_y,
                )
            previous = positions
        all_x_positions.extend(aux_positions)
        y_min = -4.75

    display_title = title or _format_task_title(str(diagram.task))
    ax.text(
        (min(all_x_positions) + max(all_x_positions)) / 2.0,
        1.88,
        display_title,
        ha="center",
        va="bottom",
        fontsize=15,
        weight="bold",
        color="#111827",
    )
    if diagram.notes:
        ax.text(
            min(all_x_positions) - 0.52,
            y_min + 0.02,
            " | ".join(diagram.notes),
            ha="left",
            va="top",
            fontsize=8.2,
            color="#4b5563",
        )

    ax.set_xlim(min(all_x_positions) - 0.72, max(all_x_positions) + 0.72)
    ax.set_ylim(y_min - 0.18, 2.12)
    ax.axis("off")
    fig.tight_layout(pad=0.45)
    return fig, ax


def save_architecture_figures(
    source: str | Path,
    output_dir: str | Path,
    *,
    formats: Sequence[str] = ("png", "svg", "pdf"),
    dpi: int = 220,
    title: str | None = None,
    basename: str = "architecture",
    include_decoder: bool = False,
) -> dict[str, str]:
    '''Save architecture figures to disk.

    Parameters
    ----------
    source : str or pathlib.Path
        Architecture JSON/YAML file or ``best_model/`` export directory.
    output_dir : str or pathlib.Path
        Directory where figure files will be written.
    formats : sequence of str, default=("png", "svg", "pdf")
        Output formats accepted by matplotlib.
    dpi : int, default=220
        Raster output resolution.
    title : str, optional
        Custom figure title.
    basename : str, default="architecture"
        Output filename stem.
    include_decoder : bool, default=False
        Whether to include the auxiliary reconstruction decoder branch.

    Returns
    -------
    dict
        Mapping from output format to absolute file path, plus ``source``.

    Raises
    ------
    ValueError
        If no valid output format is provided.
    '''

    document, architecture_path = load_architecture_document(source)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for fmt in formats:
        clean_fmt = fmt.strip().lower().lstrip(".")
        if not clean_fmt:
            continue
        fig, _ = plot_architecture_diagram(
            document, title=title, include_decoder=include_decoder
        )
        path = out_dir / f"{basename}.{clean_fmt}"
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        written[clean_fmt] = str(path.resolve())
    if not written:
        raise ValueError("At least one output format is required.")
    written["source"] = str(architecture_path.resolve())
    return written


__all__ = [
    "ArchitectureBlock",
    "ArchitectureDiagram",
    "load_architecture_document",
    "normalize_architecture",
    "plot_architecture_diagram",
    "save_architecture_figures",
]
