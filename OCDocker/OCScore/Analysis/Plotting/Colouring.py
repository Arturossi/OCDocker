#!/usr/bin/env python3

# Description
###############################################################################
'''
Color palette utilities for Analysis plots.

They are imported as:

import OCDocker.OCScore.Analysis.Plotting.Colouring as ocstatcolour
'''

# Imports
###############################################################################

import seaborn as sns
import pandas as pd



try:
    import colorcet as cc  # optional
except Exception:  # pragma: no cover
    cc = None

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

    print("Setting the pallette, alpha, and error threshold for the plots.")

    if palette_colour == "glasbey":
        if cc is None:
            # Fallback when colorcet is not available
            print("[Colouring] colorcet not available; falling back to 'tab20'.")
            palette_colour = sns.color_palette("tab20", n_colors = df['Methodology'].nunique())  # type: ignore
        else:
            palette_colour = sns.color_palette(cc.glasbey, n_colors = df['Methodology'].nunique())  # type: ignore
    elif palette_colour in ["Set2", "Set3", "tab10", "tab20", "colorblind", "pastel", "bright", "dark", "deep", "muted", "viridis"]:
        # Use seaborn's built-in palettes
        palette_colour = sns.color_palette(palette_colour, n_colors = df['Methodology'].nunique()) # type: ignore
    else:
        # User-facing error: invalid palette
        ocerror.Error.value_error(f"Unsupported palette: '{palette_colour}'. Choose from 'glasbey', 'Set2', 'Set3', 'tab10', 'tab20', 'colorblind', 'pastel', 'bright', 'dark', 'deep', 'muted', or 'viridis'.") # type: ignore
        raise ValueError(f"Unsupported palette: {palette_colour}. Choose from 'glasbey', 'Set2', 'Set3', 'tab10', 'tab20', 'colorblind', 'pastel', 'bright', 'dark', 'deep', 'muted', or 'viridis'.")

    # Create a color mapping for methodologies
    color_mapping = {
        method: color for method, color in zip(
            df['Methodology'].unique(), 
            sns.color_palette(palette_colour, n_colors = df['Methodology'].nunique())
        )
    }

    return color_mapping
