from __future__ import annotations

from collections.abc import Iterable
from difflib import get_close_matches
from re import findall, sub

import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
from matplotlib.pyplot import colormaps
from matplotlib.pyplot import get_cmap as _get_cmap

_LOWER_COLORMAPS = {cmap.lower(): cmap for cmap in colormaps()}


def get_cmap(cmap: str) -> Colormap:
    """
    Get a colormap by name.

    Args:F
        cmap: The name of the colormap.

    Returns:
        A colormap.
    """
    try:
        return _get_cmap(_LOWER_COLORMAPS[cmap.lower()])
    except KeyError:
        matches = get_close_matches(cmap, _LOWER_COLORMAPS.values(), n=5, cutoff=0.1)
        if matches:
            raise ValueError(
                f"Unknown colormap '{cmap}'. Did you mean one of these: {matches}?"
            )
        else:
            raise ValueError(f"Unknown colormap '{cmap}'.")


def cmap_to_array(
    cmap: LinearSegmentedColormap | ListedColormap | Iterable,
) -> np.ndarray:
    """
    Convert a colormap to an array of colors.

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


def sub_match(pattern: str, string: str, key: str) -> tuple[str, str]:
    """
    Find a pattern in a string and remove it.

    Args:
        pattern: The pattern to find.
        string: The string to search.
        key: The name of the pattern.

    Returns:
        The new string and the match.
    """
    matches = findall(pattern, string)
    if len(matches) > 1:
        raise ValueError(f"Should only contain one {key!r} but found {matches}")
    elif len(matches) == 1:
        string = sub(pattern, "", string, count=1)
    return string, matches[0] if matches else ""
