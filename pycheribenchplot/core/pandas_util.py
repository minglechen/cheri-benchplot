from typing import Callable, TypeVar

import pandas as pd

T = TypeVar("T")


def map_index(df: pd.DataFrame, level: str, fn: Callable[[any], any], inplace: bool = False) -> pd.DataFrame:
    """
    Convenience function to apply a mapping function to a multi-index level.

    Example:
    ```
    df = pd.DataFrame({"a": [10,20,30], "b": [100, 200, 300]}).set_index("a", append=True)
            b
      a
    0 10  100
    1 20  200
    2 30  300

    df = map_index(df, "a", lambda v: v - 1)
            b
      a
    0 9   100
    1 19  200
    2 29  300
    ```
    """
    if inplace:
        out_df = df
    else:
        out_df = df.copy()

    level_pos = df.index.names.index(level)
    if level_pos < 0:
        raise ValueError(f"Index level {level} not present in dataframe")
    if len(df.index.names) > 1:
        # MultiIndex
        out_df.index = df.index.map(lambda idx: idx[0:level_pos] + (fn(idx[level_pos]), ) + idx[level_pos + 1:])
    else:
        out_df.index = df.index.map(fn)
    return out_df


def generalized_xs(df: pd.DataFrame,
                   match: list | None = None,
                   levels: list | None = None,
                   complement=False,
                   droplevels=False,
                   **kwargs):
    """
    Generalized cross section that allows slicing on multiple named levels.
    Example:
    Given a dataframe, generaized_xs(df, [0, 1], levels=["k0", "k1"]) gives:

     k0 | k1 | k2 || V
     0  | 0  | 0  || 1
     0  | 0  | 1  || 2
     0  | 1  | 0  || 3  generalized_xs()   k0 | k1 | k2 || V
     0  | 1  | 1  || 4 ==================> 0  | 1  | 0  || 3
     1  | 0  | 0  || 5                     0  | 1  | 1  || 4
     1  | 0  | 1  || 6
     1  | 1  | 0  || 7
     1  | 1  | 1  || 8

    :param df: The dataframe to cross section
    :param match: Optional list of match values
    :param levels: Optional list of match levels, must have the same number of
    entries as ``match``
    :param complement: Return the complement of the matching rows, default False.
    :param droplevels: Whether to remove the match levels from the result, default False.
    :param \**kwargs: level=<match-value> pairs, behave as ``match`` and ``levels``
    """
    if match is None:
        match = []
    if levels is None:
        levels = []
    levels.extend(kwargs.keys())
    match.extend(kwargs.values())

    assert len(match) == len(levels)
    nlevels = len(df.index.names)
    slicer = [slice(None)] * nlevels
    for m, level_name in zip(match, levels):
        level_idx = df.index.names.index(level_name)
        slicer[level_idx] = m
    sel = pd.Series(False, index=df.index)
    sel.loc[tuple(slicer)] = True
    if complement:
        sel = ~sel
    result = df[sel]
    if droplevels:
        if nlevels > len(levels):
            result = result.droplevel(levels)
        else:
            result = result.reset_index()
    return result.copy()


def broadcast_xs(df: pd.DataFrame, chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe and a cross-section from it, with some missing index levels, generate
    the complete series or frame with the cross-section aligned to the parent frame.
    This is useful to perform an intermediate operations on a subset (e.g. the baseline frame)
    and then replicate the values for the rest of the datasets.
    """
    if df.index.nlevels > 1 and set(chunk.index.names).issubset(df.index.names):
        # First reorder the levels so that the shared levels between df and chunk are at the
        # front of df index names lis
        _, r = df.align(chunk, axis=0)
        return r.reorder_levels(df.index.names)
    else:
        if chunk.index.nlevels > 1:
            raise TypeError("Can not broadcast multiindex into flat index")
        nrepeat = len(df) / len(chunk)
        if nrepeat != int(nrepeat):
            raise TypeError("Can not broadcast non-alignable chunk")
        # Just repeat the chunk along the frame
        df = df.copy()
        df.loc[:] = chunk.values.repeat(nrepeat, axis=0)
        return df


def apply_or(fn: Callable[[any], T], default: T, *partial_args, **partial_kwargs):
    """
    Helper to apply fallible functions to a series or dataframe.

    If the function fails with an exception, the default value is returned instead.
    """
    def _inner(*args, **kwargs):
        kwargs.update(partial_kwargs)
        try:
            return fn(*(partial_args + args), **kwargs)
        except:
            return default

    return _inner
