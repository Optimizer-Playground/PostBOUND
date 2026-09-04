from __future__ import annotations

import math
from collections.abc import Callable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

Plotter = Callable[[str, pd.DataFrame, Axis], None]


def make_grid_plot(
    data: pd.DataFrame,
    *,
    plot_func: Plotter,
    label_col: str = "label",
    ncols: int = 4,
    base_widht: int = 5,
    base_height: int = 3,
) -> tuple[Figure, Axis]:
    labels = data[label_col].unique()
    nrows = math.ceil(len(labels) / ncols)
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * base_widht, nrows * base_height))
    current_col, current_row = 0, 0

    for label in labels:  # label is accessed using @ syntax
        current_ax = ax[current_row][current_col] if nrows > 1 else ax[current_col]
        current_samples = data.query(f"{label_col} == @label")

        plot_func(label, current_samples, current_ax)

        current_col = (current_col + 1) % ncols
        current_row = current_row + 1 if current_col == 0 else current_row

    extra_rows = range(ncols - len(labels) % ncols) if len(labels) % ncols != 0 else []
    for extra_col in extra_rows:
        ax[current_row][ncols - extra_col - 1].axis("off")

    return fig, ax


def make_facetted_grid_plot(
    data: pd.DataFrame,
    *,
    upper_plotter: Plotter,
    lower_plotter: Plotter,
    label_col: str = "label",
    ncols: int = 4,
    base_width: int = 5,
    base_height: int = 3,
    grid_wspace: float = 0.4,
    grid_hspace: float = 0.6,
) -> Figure:
    labels = data[label_col].unique()
    nrows = math.ceil(len(labels) / ncols)
    fig = plt.figure(constrained_layout=True, figsize=(ncols * base_width, nrows * base_height))
    parent_gridspec = GridSpec(nrows, ncols, figure=fig, wspace=grid_wspace, hspace=grid_hspace)

    for i, label in enumerate(labels):
        current_gridspec = GridSpecFromSubplotSpec(2, 1, subplot_spec=parent_gridspec[i], wspace=0.1, hspace=0.1)
        current_samples = data.query(f"{label_col} == @label")

        upper_ax = plt.Subplot(fig, current_gridspec[0])
        upper_plotter(label, current_samples, upper_ax)
        upper_ax.set_xlabel("")
        plt.setp(upper_ax.get_xticklabels(), visible=False)
        fig.add_subplot(upper_ax)

        lower_ax = plt.Subplot(fig, current_gridspec[1], sharex=upper_ax)
        lower_plotter(label, current_samples, lower_ax)
        fig.add_subplot(lower_ax)

    return fig
