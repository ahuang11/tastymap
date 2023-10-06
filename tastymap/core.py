from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    rgb2hex,
    rgb_to_hsv,
)

from .utils import cmap_to_array, get_cmap, replace_match, subset_cmap


class ColorModel(Enum):
    """Enumeration for different color models."""

    RGBA = "rgba"
    RGB = "rgb"
    HSV = "hsv"
    HEX = "hex"


class TastyMap:
    """A class to represent and manipulate colormaps in a tasty manner.

    Attributes:
        cmap (LinearSegmentedColormap): The colormap object.
        _cmap_array (np.ndarray): Array representation of the colormap.
    """

    def __init__(
        self,
        cmap: LinearSegmentedColormap,
        name: str | None = None,
        num_colors: int | None = None,
        bad: str | tuple | None = None,
        under: str | tuple | None = None,
        over: str | tuple | None = None,
    ):
        """Initializes a TastyMap instance.

        Args:
            cmap (LinearSegmentedColormap): The colormap to be used.
            name (str, optional): Name of the colormap. Defaults to the name of the provided colormap.
            num_colors (int, optional): Number of colors in the colormap. Defaults to None.
            bad (str | tuple, optional): Color for bad values. Defaults to None.
            under (str | tuple, optional): Color for underflow values. Defaults to None.
            over (str | tuple, optional): Color for overflow values. Defaults to None.
        """
        if not isinstance(cmap, LinearSegmentedColormap):
            raise TypeError(
                f"Expected LinearSegmentedColormap; received {type(cmap)!r}."
            )

        cmap = cmap.copy()
        cmap.name = name or cmap.name
        self.cmap_array = cmap_to_array(cmap)
        if num_colors is not None:
            if num_colors <= 0:
                raise ValueError("Number of colors must be positive.")
            cmap = LinearSegmentedColormap.from_list(
                cmap.name, self.cmap_array, N=num_colors
            )
        cmap.set_extremes(bad=bad, under=under, over=over)
        self.cmap = cmap.copy()

    @classmethod
    def from_str(cls, string: str, **kwargs: dict) -> TastyMap:
        """Creates a TastyMap instance from a string name.

        Args:
            string (str): Name of the colormap.
            **kwargs: Additional keyword arguments.

        Returns:
            TastyMap: A new TastyMap instance.
        """
        string, n_match = replace_match(r"_n\d+(?=_|$)", string, "_n")
        string, r_match = replace_match(r"_r+(?=_|$)", string, "_r")
        # regex to find s0 or s1,3,5,17 or s1:10 or s0:10:-1
        string, i_match = replace_match(
            r"_i(-?\d*(?:,-?\d+)*(?::\d*)?(?::-?\d+)?)", string, "_i"
        )
        cmap = get_cmap(string)  # type: ignore
        cmap_array = cmap_to_array(cmap)

        new_name = string
        if n_match:
            num_colors = int(n_match[2:])
            new_name += n_match
        else:
            num_colors = 256

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
        name: str = "custom_tastymap",
        num_colors: int | None = None,
        **kwargs: dict,
    ) -> TastyMap:
        """Creates a TastyMap instance from a list of colors.

        Args:
            colors (Iterable): List of colors.
            name (str, optional): Name of the colormap. Defaults to "custom_tastymap".
            num_colors (int, optional): Number of colors in the colormap. Defaults to 256.
            **kwargs: Additional keyword arguments.

        Returns:
            TastyMap: A new TastyMap instance.
        """
        if not colors:
            raise ValueError("Must provide at least one color.")
        cmap_array = cmap_to_array(colors)
        cmap = LinearSegmentedColormap.from_list(name, cmap_array, N=num_colors or 256)
        return TastyMap(cmap, **kwargs)

    @classmethod
    def from_listed_colormap(
        cls, listed_colormap: ListedColormap, **kwargs
    ) -> TastyMap:
        """Creates a TastyMap instance from a ListedColormap.

        Args:
            listed_colormap (ListedColormap): The colormap to be used.
            **kwargs: Additional keyword arguments.

        Returns:
            TastyMap: A new TastyMap instance.
        """
        kwargs["name"] = kwargs.get("name") or listed_colormap.name
        kwargs["num_colors"] = kwargs.get("num_colors") or listed_colormap.N
        return cls.from_list(listed_colormap.colors, **kwargs)

    def interpolate(self, num_colors: int) -> TastyMap:
        """Interpolates the colormap to a specified number of colors.

        Args:
            num_colors (int): Number of colors to interpolate to.

        Returns:
            TastyMap: A new TastyMap instance with the interpolated colormap.
        """
        cmap = LinearSegmentedColormap.from_list(
            self.cmap.name, self.cmap_array, N=num_colors
        )
        return TastyMap(cmap)

    def register(self) -> TastyMap:
        """Registers the colormap with matplotlib.

        Returns:
            TastyMap: The current TastyMap instance.
        """
        plt.register_cmap(self.cmap.name, self.cmap)
        return self

    def rename(self, name: str) -> TastyMap:
        """Renames the colormap.

        Args:
            name (str): New name for the colormap.

        Returns:
            TastyMap: A new TastyMap instance with the renamed colormap.
        """
        return TastyMap(self.cmap, name=name)

    def reverse(self) -> TastyMap:
        """Reverses the colormap.

        Returns:
            TastyMap: A new TastyMap instance with the reversed colormap.
        """
        return self[::-1]

    def to(self, color_model: ColorModel | str) -> np.ndarray:
        """Converts the colormap to a specified color model.

        Args:
            color_model (ColorModel | str): The color model to convert to.

        Returns:
            np.ndarray: Array representation of the colormap in the specified color model.
        """
        if isinstance(color_model, str):
            try:
                color_model = ColorModel(color_model.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid color model: {color_model!r}; "
                    f"select from: {[cm.value for cm in ColorModel]}."
                )

        if color_model == ColorModel.RGBA:
            return self.cmap_array
        elif color_model == ColorModel.RGB:
            return self.cmap_array[:, :3]
        elif color_model == ColorModel.HSV:
            return rgb_to_hsv(self.cmap_array[:, :3])
        elif color_model == ColorModel.HEX:
            return np.apply_along_axis(rgb2hex, 1, self.cmap_array[:, :3])

    def set(
        self,
        bad: str | tuple | None = None,
        under: str | tuple | None = None,
        over: str | tuple | None = None,
    ) -> TastyMap:
        """Sets the colors for bad, underflow, and overflow values.

        Args:
            bad (str | tuple, optional): Color for bad values. Defaults to None.
            under (str | tuple, optional): Color for underflow values. Defaults to None.
            over (str | tuple, optional): Color for overflow values. Defaults to None.

        Returns:
            TastyMap: A new TastyMap instance with the updated colormap.
        """
        return TastyMap(self.cmap, bad=bad, under=under, over=over)

    def __iter__(self):
        """Iterates over the colormap.

        Yields:
            np.ndarray: An array of colors.
        """
        yield from self.cmap_array

    def __getitem__(self, indices):
        """Gets a subset of the colormap.

        Args:
            indices: Indices to subset the colormap.

        Returns:
            TastyMap: A new TastyMap instance with the subset colormap.
        """
        cmap = subset_cmap(self.cmap, indices)
        return TastyMap(cmap)

    def _repr_html_(self) -> str:
        """Returns an HTML representation of the colormap.

        Returns:
            str: HTML representation of the colormap.
        """
        return self.cmap._repr_html_()

    def __add__(self, tmap: TastyMap) -> TastyMap:
        """Combines two TastyMap instances.

        Args:
            tmap (TastyMap): Another TastyMap instance to combine with.

        Returns:
            TastyMap: A new TastyMap instance with the combined colormap.
        """
        if not isinstance(tmap, TastyMap):
            raise TypeError(
                f"Can only combine TastyMap instances; received {type(tmap)!r}."
            )
        name = self.cmap.name + "_" + tmap.cmap.name
        cmap_array = np.concatenate([self.cmap_array, cmap_to_array(tmap.cmap)])
        num_colors = len(cmap_array)
        cmap = LinearSegmentedColormap.from_list(name, cmap_array, N=num_colors)
        return TastyMap(cmap)

    def __len__(self) -> int:
        """Returns the number of colors in the colormap.

        Returns:
            int: Number of colors in the colormap.
        """
        return len(self.cmap_array)

    def __eq__(self, other: TastyMap) -> bool:
        """Checks if two TastyMap instances are equal.

        Args:
            other (TastyMap): Another TastyMap instance to compare with.

        Returns:
            bool: True if the two TastyMap instances are equal; False otherwise.
        """
        if not isinstance(other, TastyMap):
            return False
        return np.all(self.cmap_array == other.cmap_array)

    def __str__(self) -> str:
        """Returns the name of the colormap.

        Returns:
            str: Name of the colormap.
        """
        return f"{self.cmap.name} ({len(self)} colors)"

    def __repr__(self) -> str:
        """Returns a string representation of the TastyMap instance.

        Returns:
            str: String representation of the TastyMap instance.
        """
        return f"TastyMap({self.cmap.name!r})"


def cook_tmap(
    colors_or_cmap: str | LinearSegmentedColormap | ListedColormap | Iterable,
    num_colors: int | None = None,
    reverse: bool = False,
    name: str | None = None,
    color_model: ColorModel | str | None = None,
) -> LinearSegmentedColormap | np.ndarray:
    """Cook a completely new colormap or modify an existing one.

    Args:
        colors_or_cmap: A colormap or a string name of a colormap.
        num_colors: Number of colors in the colormap. Defaults to None.
        reverse: Whether to reverse the colormap. Defaults to False.
        name (str, optional): Name of the colormap; if provided, registers the cmap. Defaults to None.
        color_model (ColorModel | str, optional): Color model to output as. Defaults to None.

    Returns:
        LinearSegmentedColormap | np.ndarray: A LinearSegmentedColormap or an array of colors.
    """
    tmap_kwargs = dict(num_colors=num_colors, name=name)
    if isinstance(colors_or_cmap, str):
        tmap = TastyMap.from_str(colors_or_cmap, **tmap_kwargs)
    elif isinstance(colors_or_cmap, LinearSegmentedColormap):
        tmap = TastyMap(colors_or_cmap, **tmap_kwargs)
    elif isinstance(colors_or_cmap, ListedColormap):
        tmap = TastyMap.from_listed_colormap(colors_or_cmap, **tmap_kwargs)
    elif isinstance(colors_or_cmap, Iterable):
        tmap = TastyMap.from_list(colors_or_cmap, **tmap_kwargs)
    else:
        raise TypeError(
            f"Expected str, LinearSegmentedColormap, ListedColormap, or Iterable; "
            f"received {type(colors_or_cmap)!r}."
        )

    if reverse:
        tmap = tmap.reverse()
    if name:
        tmap = tmap.rename(name)
        tmap.register()
    if color_model:
        tmap = tmap.to(color_model)

    return tmap
