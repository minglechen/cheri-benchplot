from dataclasses import dataclass
from itertools import chain
from typing import GenericAlias

import numpy as np
import pandas as pd
import squarify as sq
from matplotlib.patches import Rectangle
from matplotlib.pyplot import get_cmap
from matplotlib.text import Text

BBox = GenericAlias(tuple, (float, ) * 4)


@dataclass
class TreemapArgs:
    ax: "Axes"
    values: str
    groups: str
    basecolor: tuple | None
    threshold: float
    maxlevel: int


def _mosaic_treemap_level(level: int, df: pd.DataFrame, bbox: BBox, args: TreemapArgs) -> pd.DataFrame:
    """
    Draw a level of the treemap mosaic

    :return: A tuple containing a list of rectangles and a list of text
    """
    EMPTY = ([], [])
    x, y, width, height = bbox
    if level >= args.maxlevel:
        return EMPTY

    # Determine what is part of the next level
    df["_treemap_level"] = df[args.groups].map(lambda g: g[level] if len(g) > level else None)
    df = df.dropna(subset="_treemap_level")

    grouped = df.groupby("_treemap_level").sum().sort_values(by=[args.values], ascending=False)
    if len(grouped) < 2:
        # We are done
        return EMPTY

    norm_values = sq.normalize_sizes(grouped[args.values], width, height)
    rects = sq.squarify(norm_values, x, y, width, height)
    rects_df = pd.DataFrame(rects)
    # add padding to rectangles, the padding is scaled by width and height so that
    # it is always 0.5% of the dimension.
    # If a rectangle is too small, skip padding.
    pad_x = width / 200
    pad_y = height / 200
    if (rects_df["dx"] - pad_x > 0).all() and (rects_df["dy"] - pad_y > 0).all():
        rects_df["x"] += pad_x / 2
        rects_df["dx"] -= pad_x
        rects_df["y"] += pad_y / 2
        rects_df["dy"] -= pad_y

    # Now we can draw the rectangles
    show_df = pd.concat([grouped.reset_index(), rects_df], axis=1)
    cmap = get_cmap()
    colors = cmap(np.linspace(0, 1, num=len(show_df)))
    show_df["color"] = list(map(tuple, colors))

    def mkrect(r):
        return Rectangle((r["x"], r["y"]), r["dx"], r["dy"], color=r["color"])

    rects = show_df.apply(mkrect, axis=1)

    def mktext(r):
        center_x = r["x"] + r["dx"] / 2
        center_y = r["y"] + r["dy"] / 2
        return Text(center_x,
                    center_y,
                    text=r["_treemap_level"],
                    ha="center",
                    va="center",
                    alpha=(level + 1) / args.maxlevel)

    text = show_df.apply(mktext, axis=1)

    for r, t in zip(rects, text):
        args.ax.add_patch(r)
        args.ax.add_artist(t)

    # Now, recursively enter the rectangles for each group
    def next_level(row):
        next_df = df.loc[df["_treemap_level"] == row["_treemap_level"]].copy()
        row_bbox = (row["x"], row["y"], row["dx"], row["dy"])
        return _mosaic_treemap_level(level + 1, next_df, row_bbox, args)

    result = show_df.apply(next_level, axis=1)
    sub_rects = chain.from_iterable(result.map(lambda v: v[0]))
    sub_text = chain.from_iterable(result.map(lambda v: v[1]))
    rects = rects.tolist() + list(sub_rects)
    text = text.tolist() + list(sub_text)

    return rects, text


def mosaic_treemap(df: pd.DataFrame,
                   fig: "Figure",
                   ax: "Axes",
                   bbox: BBox,
                   values: str,
                   groups: str,
                   threshold: float = 0,
                   maxlevel: int = 0):
    """
    Produce a hierarchical treemap plot in the initial bounding box.

    :param df: Input dataframe
    :param ax: Matplotlib axes
    :param bbox: Tuple containing (x, y, dx, dy)
    :param values: Name of the dataframe column to use for values
    :param groups: How to groups the dataframe to produce the values
    for the current tree level. It is assumed that the values should be added
    across the group.
    :param next_fn: Callable that transforms the dataframe to show the
    grouping for the next level of the hierarchy.
    """
    x, y, width, height = bbox
    if maxlevel == 0:
        maxlevel = df[groups].map(len).max()

    args = TreemapArgs(ax=ax, values=values, groups=groups, basecolor=None, threshold=threshold, maxlevel=maxlevel)
    # Handle a treemap level at a time
    rects, text = _mosaic_treemap_level(0, df, bbox, args)

    # Adjust axes
    ax.set_xlim(x, x + width)
    ax.set_ylim(y, y + height)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    # Adjust text size, only ever take up to 95% of the box
    fig.draw_without_rendering()
    for r, t in zip(rects, text):
        ext_r = r.get_window_extent()
        ext_t = t.get_window_extent()
        scale = min(ext_r.width / ext_t.width, ext_r.height / ext_t.height) * 0.95
        t.set_size(t.get_size() * scale)
