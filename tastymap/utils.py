from __future__ import annotations

from collections.abc import Sequence
from difflib import get_close_matches
from re import IGNORECASE, findall, sub

import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
from matplotlib.pyplot import colormaps
from matplotlib.pyplot import get_cmap as _get_cmap


def get_registered_cmaps() -> dict[str, str]:
    """
    Get a mapping of registered colormaps.

    Returns:
        A mapping of registered colormaps.
    """
    return {cmap.lower(): cmap for cmap in colormaps()}


def get_cmap(cmap: str) -> Colormap:
    """
    Get a colormap by name.

    Args:
        cmap: The name of the colormap.

    Returns:
        A colormap.
    """
    lower_colormaps = get_registered_cmaps()
    try:
        return _get_cmap(lower_colormaps[cmap.lower()])
    except KeyError:
        matches = get_close_matches(cmap, lower_colormaps.values(), n=5, cutoff=0.1)
        if matches:
            raise ValueError(
                f"Unknown colormap '{cmap}'. Did you mean one of these: {matches}?"
            )
        else:
            raise ValueError(f"Unknown colormap '{cmap}'.")


def subset_cmap(
    cmap: Colormap,
    indices: int | float | slice | Sequence,
    name: str | None = None,
) -> LinearSegmentedColormap:
    """
    Subset a colormap.

    Args:
        cmap: A colormap.
        indices: The indices to subset.
        name: The name of the new colormap.

    Returns:
        A new colormap.
    """
    cmap_array = cmap_to_array(cmap)
    name = name or cmap.name
    if isinstance(indices, (int, float)):
        cmap_indices = np.array([indices] * 2).astype(int)
        name += f"_i{indices}"
    elif isinstance(indices, Sequence):
        cmap_indices = np.array(indices)
        if len(cmap_indices) == 1:
            cmap_indices = np.array([cmap_indices] * 2).astype(int)
        name += f"_i{','.join(str(i) for i in cmap_indices.flat)}"
    elif isinstance(indices, slice):
        cmap_indices = indices  # type: ignore
        step = indices.step
        start = indices.start
        stop = indices.stop
        if not indices.start and not indices.stop:
            if step == -1:
                name, r_match = replace_match(r"_r+(?=_|$)", name, "_r")
                if not r_match:
                    name += "_r"
            else:
                name += f"_i::{step}"
        else:
            name += f"_i{start}:{stop}:{step}"

    cmap_array = cmap_array[cmap_indices]
    cmap = LinearSegmentedColormap.from_list(name, cmap_array, N=len(cmap_array))
    return cmap


def cmap_to_array(
    cmap: Colormap | Sequence,
) -> np.ndarray:
    """
    Convert a colormap to an array of colors as RGB.

    Args:
        cmap: A colormap.

    Returns:
        An array of colors.
    """
    if isinstance(cmap, str):
        cmap = get_cmap(cmap)  # type: ignore

    if isinstance(cmap, LinearSegmentedColormap):
        cmap_array = cmap(np.linspace(0, 1, cmap.N))
    elif isinstance(cmap, ListedColormap):
        cmap_array = np.array(cmap.colors)
    else:
        cmap_array = np.array(cmap)
    return cmap_array


def replace_match(pattern: str, string: str, key: str) -> tuple[str, str]:
    """
    Find a pattern in a string and remove it.

    Args:
        pattern: The pattern to find.
        string: The string to search.
        key: The name of the pattern.

    Returns:
        The new string and the match.
    """
    matches = findall(pattern, string, IGNORECASE)
    if len(matches) > 1:
        raise ValueError(f"Should only contain one {key!r} but found {matches}")
    elif len(matches) == 1:
        string = sub(pattern, "", string, count=1, flags=IGNORECASE)
    return string, matches[0] if matches else ""
