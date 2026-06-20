#!/usr/bin/env python3

# Description
###############################################################################
'''
Color palette utilities for Analysis plots.

Usage:

import OCDocker.OCScore.Analysis.Plotting.Colouring as ocstatcolour
'''

# Imports
###############################################################################
import pandas as pd
import seaborn as sns
import OCDocker.Error as ocerror
import OCDocker.Toolbox.Logging as oclogging

LOGGER = oclogging.get_logger("ocscore.analysis.plotting.colouring")

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

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

try:
    import colorcet as cc  # optional
except Exception:  # pragma: no cover
    cc = None

## Public ##


def set_color_mapping(df: pd.DataFrame, palette_colour: str = "glasbey") -> dict[str, tuple[float, float, float]]:
    '''
    Set the color palette for plotting based on the unique methodologies in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'Methodology' column with unique methodologies.
    palette_colour : str
        Name of the color palette to use. Options include:
        - "glasbey"
        - "Set2"
        - "Set3"
        - "tab10"
        - "tab20"
        - "colorblind"
        - "pastel"
        - "bright"
        - "dark"
        - "deep"
        - "muted"
        - "viridis"

    Returns
    -------
    color_mapping : dict[str, tuple[float, float, float]]
        Dictionary mapping each methodology to a color in RGB format.

    Raises
    ------
    ValueError
        If an unsupported palette is provided.
    '''

    LOGGER.info("Setting colour palette for plots.")

    if palette_colour == "glasbey":
        if cc is None:
            # Fallback when colorcet is not available
            LOGGER.warning("colorcet not available; falling back to 'tab20'.")
            palette_colour = sns.color_palette("tab20", n_colors = df['Methodology'].nunique())
        else:
            palette_colour = sns.color_palette(cc.glasbey, n_colors = df['Methodology'].nunique())
    elif palette_colour in ["Set2", "Set3", "tab10", "tab20", "colorblind", "pastel", "bright", "dark", "deep", "muted", "viridis"]:
        # Use seaborn's built-in palettes
        palette_colour = sns.color_palette(palette_colour, n_colors = df['Methodology'].nunique())
    else:
        # User-facing error: invalid palette
        ocerror.Error.value_error(f"Unsupported palette: '{palette_colour}'. Choose from 'glasbey', 'Set2', 'Set3', 'tab10', 'tab20', 'colorblind', 'pastel', 'bright', 'dark', 'deep', 'muted', or 'viridis'.")
        raise ValueError(f"Unsupported palette: {palette_colour}. Choose from 'glasbey', 'Set2', 'Set3', 'tab10', 'tab20', 'colorblind', 'pastel', 'bright', 'dark', 'deep', 'muted', or 'viridis'.")

    # Create a color mapping for methodologies
    color_mapping = {
        method: color for method, color in zip(
            df['Methodology'].unique(),
            sns.color_palette(palette_colour, n_colors = df['Methodology'].nunique())
        )
    }

    return color_mapping
