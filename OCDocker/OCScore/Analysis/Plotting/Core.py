
from __future__ import annotations
import matplotlib.pyplot as plt

def apply_basic_style():
    plt.rcParams.update({
        "figure.autolayout": True,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

def new_fig(size=(6,4)):
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    return fig, ax
