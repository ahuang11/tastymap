from __future__ import annotations

from typing import Any
from collections.abc import Sequence

from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap

from tastymap.models import ColorModel, TastyColorMap, MatplotlibTastyColorBar


def cook_tcmap(
    colors_or_cmap: Sequence | str | Colormap,
    num_colors: int | None = None,
    reverse: bool = False,
    name: str | None = None,
    bad: str | tuple | None = None,
    under: str | tuple | None = None,
    over: str | tuple | None = None,
    from_color_model: ColorModel | str | None = None,
) -> TastyColorMap:
    """Cook a completely new colormap or modify an existing one.

    Args:
        colors_or_cmap: A list of colors or a colormap instance or string.
        num_colors: Number of colors in the colormap. Defaults to None.
        reverse: Whether to reverse the colormap. Defaults to False.
        name: Name of the colormap; if provided, registers the cmap. Defaults to None.
        bad: Color for bad values. Defaults to None.
        under: Color for underflow values. Defaults to None.
        over: Color for overflow values. Defaults to None.
        from_color_model: Color model of the input colormap; required if
            colors_or_cmap is an Sequence and not hexcodes. Defaults to None.

    Returns:
        TastyColorMap: A new TastyColorMap instance with the new colormap.
    """
    if isinstance(colors_or_cmap, str):
        tmap = TastyColorMap.from_str(colors_or_cmap)
    elif isinstance(colors_or_cmap, LinearSegmentedColormap):
        tmap = TastyColorMap(colors_or_cmap)
    elif isinstance(colors_or_cmap, ListedColormap):
        tmap = TastyColorMap.from_listed_colormap(colors_or_cmap)
    elif isinstance(colors_or_cmap, Sequence):
        if not isinstance(colors_or_cmap[0], str) and from_color_model is None:
            raise ValueError(
                "Please specify from_color_model to differentiate "
                "between RGB and HSV color models."
            )
        tmap = TastyColorMap.from_list(
            colors_or_cmap, color_model=from_color_model or ColorModel.RGB
        )
    else:
        raise TypeError(
            f"Expected str, LinearSegmentedColormap, ListedColormap, or Sequence; "
            f"received {type(colors_or_cmap)!r}."
        )

    if bad or under or over:
        tmap = tmap.set_extremes(bad=bad, under=under, over=over)

    if num_colors:
        tmap = tmap.resize(num_colors)

    if reverse:
        tmap = tmap.reverse()

    if name:
        tmap.cmap.name = name
        tmap.register()

    return tmap


def pair_tcbar(
    plot: Any,
    colors_or_cmap_or_tcmap: (Sequence | str | Colormap | TastyColorMap),
    bounds: tuple[float, float] | Sequence[float],
    labels: list[str] | None = None,
    uniform_spacing: bool = True,
    **tcbar_kwargs,
):
    tcmap = colors_or_cmap_or_tcmap
    if not isinstance(tcmap, TastyColorMap):
        tcmap = cook_tcmap(colors_or_cmap_or_tcmap)

    if hasattr(plot, "axes"):
        tcbar = MatplotlibTastyColorBar(
            tcmap,
            bounds=bounds,
            labels=labels,
            uniform_spacing=uniform_spacing,
            **tcbar_kwargs,
        )
    else:
        raise NotImplementedError("Only matplotlib plots are supported.")
    return tcbar.add_to(plot)
