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

from .utils import cmap_to_array, get_cmap, sub_match, subset_cmap


class TastyMap:
    def __init__(
        self,
        cmap: LinearSegmentedColormap,
        bad: str | tuple | None = None,
        under: str | tuple | None = None,
        over: str | tuple | None = None,
    ):
        self._cmap = cmap.copy()
        self._cmap.set_extremes(bad=bad, under=under, over=over)

    @classmethod
    def from_str(cls, name: str, **kwargs: dict) -> TastyMap:
        name, n_match = sub_match(r"_n\d+(?=_|$)", name, "_n")
        name, r_match = sub_match(r"_r+(?=_|$)", name, "_r")
        # regex to find s0 or s1,3,5,17 or s1:10 or s0:10:-1
        name, i_match = sub_match(r"_i(-?\d*(?:,-?\d+)*(?::\d*)?(?::-?\d+)?)", name, "_i")
        cmap = get_cmap(name)  # type: ignore
        cmap_array = cmap_to_array(cmap)

        new_name = name
        if n_match:
            num_colors = int(n_match[2:])
            new_name += n_match
        else:
            num_colors = 255

        if r_match:
            cmap_array = cmap_array[::-1]
            new_name += r_match

        cmap = LinearSegmentedColormap.from_list(new_name, cmap_array, N=num_colors)
        if i_match:
            if "," in i_match:
                cmap_indices = np.array(i_match.split(",")).astype(int)
            elif ":" in i_match:
                start, stop = i_match.split(":", maxsplit=1)
                if ":" in stop:
                    stop, step = stop.split(":", maxsplit=1)
                else:
                    step = 1
                start = int(start) if start else None
                stop = int(stop) if stop else None
                step = int(step) if step else None
                cmap_indices = slice(start, stop, step)
            else:
                cmap_indices = int(i_match)
            cmap = subset_cmap(cmap, cmap_indices, new_name)

        return TastyMap(cmap, **kwargs)

    @classmethod
    def from_list(
        cls,
        colors: Iterable,
        name: str = "tastymap",
        num_colors: int | None = None,
        **kwargs: dict,
    ) -> TastyMap:
        cmap_array = cmap_to_array(colors)
        cmap = LinearSegmentedColormap.from_list(name, cmap_array, N=num_colors)
        return TastyMap(cmap, **kwargs)

    @classmethod
    def from_listed_colormap(
        cls, listed_colormap: ListedColormap, **kwargs
    ) -> TastyMap:
        return cls.from_list(
            listed_colormap.colors, listed_colormap.name, listed_colormap.N, **kwargs
        )

    def interpolate(self, num_colors: int) -> TastyMap:
        cmap_array = cmap_to_array(self._cmap)
        cmap = LinearSegmentedColormap.from_list(
            self._cmap.name, cmap_array, N=num_colors
        )
        return TastyMap(cmap)

    def register(self, name: str | None = None) -> TastyMap:
        plt.register_cmap(name or self._cmap.name, self._cmap)
        return self

    def rename(self, name: str) -> TastyMap:
        self._cmap.name = name
        return self

    def __getitem__(self, indices):
        cmap = subset_cmap(self._cmap, indices)
        return TastyMap(cmap)

    def _repr_html_(self) -> str:
        return self._cmap._repr_html_()

    def __add__(self, tasty_map: TastyMap) -> TastyMap:
        name = self._cmap.name + "_" + tasty_map._cmap.name
        cmap_array = np.concatenate(
            [cmap_to_array(self._cmap), cmap_to_array(tasty_map._cmap)]
        )
        cmap = LinearSegmentedColormap.from_list(name, cmap_array)
        return TastyMap(cmap)
