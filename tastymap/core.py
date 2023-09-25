from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    rgb2hex,
    rgb_to_hsv,
)

from .utils import cmap_to_array, get_cmap, sub_match


class ColorModel(Enum):
    RGBA = "rgba"
    RGB = "rgb"
    HSV = "hsv"
    HEX = "hex"


def cook_cmap(
    colors_or_cmap: LinearSegmentedColormap | ListedColormap | Iterable,
    num_colors: int | None = None,
    reverse: bool | None = None,
    name: str | None = None,
    color_model: ColorModel | Literal["rgba", "rgb", "hsv", "hex"] | None = None,
) -> LinearSegmentedColormap | np.ndarray:
    """
    Args:
        colors_or_cmap: A list of colors or a colormap. Can use existing
            matplotlib colormaps by name and append "_n<num_colors>" to the
            name to specify the number of colors in the colormap. Can also
            append "_r" to reverse the colormap. These flags can be combined
            and take precedence over the `num_colors` and `reverse` arguments.
        num_colors: The number of colors in the colormap.
        reverse: Whether to reverse the colormap.
        name: The name of the colormap; if provided, registers the colormap
            so that it can be used by providing the name to pyplot functions.
        color_model: The color model to output the colormap in; if provided
            outputs an np.ndarray instead of a LinearSegmentedColormap.

    Returns:
        A colormap.
    """
    default_name = "tastemap"
    if isinstance(colors_or_cmap, str):
        colors_or_cmap = colors_or_cmap.lower()
        colors_or_cmap, n_match = sub_match(r"_n\d+(?=_|$)", colors_or_cmap, "_n")
        colors_or_cmap, r_match = sub_match("_r+(?=_|$)", colors_or_cmap, "_r")
        colors_or_cmap = get_cmap(colors_or_cmap)  # type: ignore
        default_name = name or colors_or_cmap.name  # type: ignore
        if n_match:
            num_colors = int(n_match[2:])
            default_name += n_match
        if r_match:
            reverse = True
            default_name += r_match

    if num_colors is None and hasattr(colors_or_cmap, "N"):
        num_colors = colors_or_cmap.N
    else:
        num_colors = num_colors or 256

    cmap_array = cmap_to_array(colors_or_cmap)
    if reverse:
        cmap_array = cmap_array[::-1]

    cmap = LinearSegmentedColormap.from_list(
        name or default_name, cmap_array, N=num_colors
    )

    if name:
        plt.register_cmap(name, cmap)

    if color_model is not None:
        cmap_array = cmap_to_array(cmap)  # outputs RGBA
        if isinstance(color_model, str):
            color_model = ColorModel(color_model.lower())

        if color_model == ColorModel.RGB:
            return cmap_array[:, :3]
        elif color_model == ColorModel.HEX:
            return np.array([rgb2hex(c) for c in cmap_array])
        elif color_model == ColorModel.HSV:
            return rgb_to_hsv(cmap_array[:, :3])
    return cmap
