#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-07 11:44:22 (ywatanabe)"

# https://stackoverflow.com/questions/53872439/half-not-split-violin-plots-in-seaborn

import seaborn as sns
from matplotlib import pyplot as plt


# def categorical_kde_plot(
#     df,
#     variable,
#     category,
#     category_order=None,
#     horizontal=False,
#     rug=True,
#     figsize=None,
# ):
#     """Draw a categorical KDE plot

#     Parameters
#     ----------
#     df: pd.DataFrame
#         The data to plot
#     variable: str
#         The column in the `df` to plot (continuous variable)
#     category: str
#         The column in the `df` to use for grouping (categorical variable)
#     horizontal: bool
#         If True, draw density plots horizontally. Otherwise, draw them
#         vertically.
#     rug: bool
#         If True, add also a sns.rugplot.
#     figsize: tuple or None
#         If None, use default figsize of (7, 1*len(categories))
#         If tuple, use that figsize. Given to plt.subplots as an argument.
#     """
#     if category_order is None:
#         categories = list(df[category].unique())
#     else:
#         categories = category_order[:]

#     figsize = (7, 1.0 * len(categories))

#     fig, axes = plt.subplots(
#         nrows=len(categories) if horizontal else 1,
#         ncols=1 if horizontal else len(categories),
#         figsize=figsize[::-1] if not horizontal else figsize,
#         sharex=horizontal,
#         sharey=not horizontal,
#     )

#     for i, (cat, ax) in enumerate(zip(categories, axes)):
#         sns.kdeplot(
#             data=df[df[category] == cat],
#             x=variable if horizontal else None,
#             y=None if horizontal else variable,
#             # kde kwargs
#             bw_adjust=0.5,
#             clip_on=False,
#             fill=True,
#             alpha=1,
#             linewidth=1.5,
#             ax=ax,
#             color="lightslategray",
#         )

#         keep_variable_axis = (i == len(fig.axes) - 1) if horizontal else (i == 0)

#         if rug:
#             sns.rugplot(
#                 data=df[df[category] == cat],
#                 x=variable if horizontal else None,
#                 y=None if horizontal else variable,
#                 ax=ax,
#                 color="black",
#                 height=0.025 if keep_variable_axis else 0.04,
#             )

#         _format_axis(
#             ax,
#             cat,
#             horizontal,
#             keep_variable_axis=keep_variable_axis,
#         )

#     plt.tight_layout()
#     plt.show()

def categorical_kde_plot(
    df,
    variable,
    category,
    category_order=None,
    horizontal=False,
    rug=True,
    figsize=None,
    ax=None,
):
    """Draw a categorical KDE plot

    Parameters
    ----------
    df: pd.DataFrame
        The data to plot
    variable: str
        The column in the `df` to plot (continuous variable)
    category: str
        The column in the `df` to use for grouping (categorical variable)
    horizontal: bool
        If True, draw density plots horizontally. Otherwise, draw them
        vertically.
    rug: bool
        If True, add also a sns.rugplot.
    figsize: tuple or None
        If None, use default figsize of (7, 1*len(categories))
        If tuple, use that figsize. Given to plt.subplots as an argument.
    """
    if category_order is None:
        categories = list(df[category].unique())
    else:
        categories = category_order[:]

    figsize = (7, 1.0 * len(categories))

    if ax is None:
    fig, axes = plt.subplots(
        nrows=len(categories) if horizontal else 1,
        ncols=1 if horizontal else len(categories),
        figsize=figsize[::-1] if not horizontal else figsize,
        sharex=horizontal,
        sharey=not horizontal,
    )

    for i, (cat, ax) in enumerate(zip(categories, axes)):
        sns.kdeplot(
            data=df[df[category] == cat],
            x=variable if horizontal else None,
            y=None if horizontal else variable,
            # kde kwargs
            bw_adjust=0.5,
            clip_on=False,
            fill=True,
            alpha=1,
            linewidth=1.5,
            ax=ax,
            color="lightslategray",
        )

        keep_variable_axis = (i == len(fig.axes) - 1) if horizontal else (i == 0)

        if rug:
            sns.rugplot(
                data=df[df[category] == cat],
                x=variable if horizontal else None,
                y=None if horizontal else variable,
                ax=ax,
                color="black",
                height=0.025 if keep_variable_axis else 0.04,
            )

        _format_axis(
            ax,
            cat,
            horizontal,
            keep_variable_axis=keep_variable_axis,
        )

    plt.tight_layout()
    plt.show()
    

def _format_axis(ax, category, horizontal=False, keep_variable_axis=True):

    # Remove the axis lines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if horizontal:
        ax.set_ylabel(None)
        lim = ax.get_ylim()
        ax.set_yticks([(lim[0] + lim[1]) / 2])
        ax.set_yticklabels([category])
        if not keep_variable_axis:
            ax.get_xaxis().set_visible(False)
            ax.spines["bottom"].set_visible(False)
    else:
        ax.set_xlabel(None)
        lim = ax.get_xlim()
        ax.set_xticks([(lim[0] + lim[1]) / 2])
        ax
