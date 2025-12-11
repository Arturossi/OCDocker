
#!/usr/bin/env python3

# Description
###############################################################################
'''
Matplotlib styling and single-axes figure helper for Analysis plots.
'''

# Imports
###############################################################################

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

# Methods
###############################################################################


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
