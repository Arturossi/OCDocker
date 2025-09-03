import seaborn as sns
import pandas as pd
import colorcet as cc

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
        palette_colour = sns.color_palette(cc.glasbey, n_colors = df['Methodology'].nunique()) # type: ignore
    elif palette_colour in ["Set2", "Set3", "tab10", "tab20", "colorblind", "pastel", "bright", "dark", "deep", "muted", "viridis"]:
        # Use seaborn's built-in palettes
        palette_colour = sns.color_palette(palette_colour, n_colors = df['Methodology'].nunique()) # type: ignore
    else:
        raise ValueError(f"Unsupported palette: {palette_colour}. Choose from 'glasbey', 'Set2', 'Set3', 'tab10', 'tab20', 'colorblind', 'pastel', 'bright', 'dark', 'deep', 'muted', or 'viridis'.")

    # Create a color mapping for methodologies
    color_mapping = {
        method: color for method, color in zip(
            df['Methodology'].unique(), 
            sns.color_palette(palette_colour, n_colors = df['Methodology'].nunique())
        )
    }

    return color_mapping
