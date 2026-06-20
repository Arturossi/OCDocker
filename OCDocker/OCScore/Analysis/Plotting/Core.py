#!/usr/bin/env python3

# Description
###############################################################################
'''
Matplotlib styling and single-axes figure helper for Analysis plots.

Usage:

from OCDocker.OCScore.Analysis.Plotting import Core as ocplotcore
'''

# Imports
###############################################################################
from __future__ import annotations

import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Tuple

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


def apply_basic_style() -> None:
    '''Apply a lightweight, consistent Matplotlib style for analysis plots.'''

    plt.rcParams.update({
        "figure.autolayout": True,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,


    })


def new_fig(size: Tuple[float, float] = (6, 4)) -> Tuple[Figure, Axes]:
    '''Create a new figure and a single axes with the standard style.

    Parameters
    ----------
    size : tuple(float, float), optional
        Figure size (width, height) in inches. Default: (6, 4).

    Returns
    -------
    (Figure, Axes)
        Newly created figure and axes.
    '''
    
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    return fig, ax
