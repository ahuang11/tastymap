from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    hsv_to_rgb,
    rgb2hex,
    rgb_to_hsv,
)

from .utils import cmap_to_array, get_cmap, subset_cmap


class ColorModel(Enum):
    """Enumeration for different color models."""

    RGBA = "rgba"
    RGB = "rgb"
    HSV = "hsv"
    HEX = "hex"


class TastyMap:
    """A class to represent and manipulate colormaps in a tasty manner.

    Attributes:
        cmap: The colormap object.
        _cmap_array: RGB array representation of the colormap.
    """

    def __init__(
        self,
        cmap: LinearSegmentedColormap,
        name: str | None = None,
    ):
        """Initializes a TastyMap instance.

        Args:
            cmap: The colormap to be used.
            name: Name of the colormap. Defaults to the name of the provided colormap.
        """
        if not isinstance(cmap, LinearSegmentedColormap):
            raise TypeError(
                f"Expected LinearSegmentedColormap; received {type(cmap)!r}."
            )

        cmap = cmap.copy()
        cmap.name = name or cmap.name
        self.cmap: LinearSegmentedColormap = cmap
        self._cmap_array = cmap_to_array(cmap)

    @classmethod
    def from_str(cls, string: str, **kwargs: dict) -> TastyMap:
        """Creates a TastyMap instance from a string name.

        Args:
            string: Name of the colormap.
            **kwargs: Additional keyword arguments.

        Returns:
            TastyMap: A new TastyMap instance.
        """
        cmap = get_cmap(string)  # type: ignore
        _cmap_array = cmap_to_array(cmap)

        new_name = string
        cmap = LinearSegmentedColormap.from_list(
            new_name, _cmap_array, N=len(_cmap_array)
        )
        return TastyMap(cmap, **kwargs)

    @classmethod
    def from_list(
        cls,
        colors: Iterable,
        name: str = "custom_tastymap",
        color_model: ColorModel | str = ColorModel.RGBA,
    ) -> TastyMap:
        """Creates a TastyMap instance from a list of colors.

        Args:
            colors: List of colors.
            name: Name of the colormap. Defaults to "custom_tastymap".

        Returns:
            TastyMap: A new TastyMap instance.
        """
        if not isinstance(colors, Iterable):
            raise TypeError(
                f"Expected Iterable; received {type(colors)!r}."
            )
        if colors is None or len(colors) == 0:
            raise ValueError("Must provide at least one color.")
        _cmap_array = cmap_to_array(colors)

        if color_model in (ColorModel.HSV, ColorModel.HSV.value):
            _cmap_array = hsv_to_rgb(_cmap_array)

        cmap = LinearSegmentedColormap.from_list(name, _cmap_array, N=len(_cmap_array))
        return TastyMap(cmap)

    @classmethod
    def from_listed_colormap(
        cls,
        listed_colormap: ListedColormap,
        name: str = "custom_tastymap",
    ) -> TastyMap:
        """Creates a TastyMap instance from a ListedColormap.

        Args:
            listed_colormap: The colormap to be used.
            name: Name of the colormap. Defaults to "custom_tastymap".

        Returns:
            TastyMap: A new TastyMap instance.
        """
        return cls.from_list(listed_colormap.colors, name=name)

    def interpolate(self, num_colors: int) -> TastyMap:
        """Interpolates the colormap to a specified number of colors.

        Args:
            num_colors: Number of colors to interpolate to.

        Returns:
            TastyMap: A new TastyMap instance with the interpolated colormap.
        """
        cmap = LinearSegmentedColormap.from_list(
            self.cmap.name, self._cmap_array, N=num_colors
        )
        return TastyMap(cmap)

    def register(self, name: str | None = None, echo: bool = True) -> TastyMap:
        """Registers the colormap with matplotlib.

        Returns:
            TastyMap: A new TastyMap instance with the registered colormap.
        """
        tmap = self.rename(name) if name else self
        colormaps.register(self.cmap, name=tmap.cmap.name, force=True)
        if echo:
            print(
                f"Sucessfully registered the colormap; "
                f"to use, set `cmap={tmap.cmap.name!r}` in your plot."
            )
        return tmap

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
            color_model: The color model to convert to.

        Returns:
            np.ndarray: Array representation of the colormap
                in the specified color model.
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
            return self._cmap_array
        elif color_model == ColorModel.RGB:
            return self._cmap_array[:, :3]
        elif color_model == ColorModel.HSV:
            return rgb_to_hsv(self._cmap_array[:, :3])
        elif color_model == ColorModel.HEX:
            return np.apply_along_axis(rgb2hex, 1, self._cmap_array[:, :3])

    def set(
        self,
        bad: str | tuple | None = None,
        under: str | tuple | None = None,
        over: str | tuple | None = None,
    ) -> TastyMap:
        """Sets the colors for bad, underflow, and overflow values.

        Args:
            bad: Color for bad values. Defaults to None.
            under: Color for underflow values. Defaults to None.
            over: Color for overflow values. Defaults to None.

        Returns:
            TastyMap: A new TastyMap instance with the updated colormap.
        """
        cmap = self.cmap.copy()
        cmap.set_extremes(bad=bad, under=under, over=over)
        return TastyMap(cmap)

    def tweak(
        self,
        hue: float | None = None,
        saturation: float | None = None,
        value: float | None = None,
        name: str | None = None,
    ) -> TastyMap:
        """Tweaks the hue, saturation, and value of the colormap.

        Args:
            hue: Hue factor (-255 to 255) to tweak by.
            saturation: Saturation factor (-10 to 10) to tweak by.
            value: Brightness value factor (0, 3) to tweak by.
            name: Name of the new colormap.

        Returns:
            TastyMap: A new TastyMap instance with the tweaked colormap.
        """
        _cmap_array = self._cmap_array.copy()
        _cmap_array[:, :3] = rgb_to_hsv(_cmap_array[:, :3])
        if hue is not None:
            hue /= 255
            if abs(hue) > 1:
                raise ValueError("Hue must be between -255 and 255 (non-inclusive).")
            _cmap_array[:, 0] = (_cmap_array[:, 0] + hue) % 1
        if saturation is not None:
            if abs(saturation) > 10:
                raise ValueError("Saturation must be between -10 and 10.")
            _cmap_array[:, 1] = _cmap_array[:, 1] * saturation
        if value is not None:
            if value < 0 or value > 3:
                raise ValueError("Value must be between 0 and 3.")
            _cmap_array[:, 2] = _cmap_array[:, 2] * value
        _cmap_array[:, :3] = hsv_to_rgb(np.clip(_cmap_array[:, :3], 0, 1))
        cmap = LinearSegmentedColormap.from_list(
            name or self.cmap.name, _cmap_array, N=len(_cmap_array)
        )
        return TastyMap(cmap)

    def __iter__(self):
        """Iterates over the colormap.

        Yields:
            np.ndarray: An array of colors.
        """
        yield from self._cmap_array

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

    def __add__(self, hue: float) -> TastyMap:
        """Adds a hue factor to the colormap.

        Args:
            hue: Hue factor to add.

        Returns:
            TastyMap: A new TastyMap instance with the hue
                added to the colormap.
        """
        return self.tweak(hue=hue)

    def __sub__(self, hue: float) -> TastyMap:
        """Subtracts a hue factor to the colormap.

        Args:
            hue: Hue factor to subtract.

        Returns:
            TastyMap: A new TastyMap instance with the hue
                subtracted from the colormap.
        """
        return self.tweak(hue=-hue)

    def __mul__(self, saturation: float) -> TastyMap:
        """Multiplies a saturation factor to the colormap.

        Args:
            saturation: Saturation factor to multiply.

        Returns:
            TastyMap: A new TastyMap instance with the saturation
                multiplied to the colormap.
        """
        return self.tweak(saturation=saturation)

    def __truediv__(self, saturation: float) -> TastyMap:
        """Divides a saturation factor to the colormap.

        Args:
            saturation: Saturation factor to divide.

        Returns:
            TastyMap: A new TastyMap instance with the saturation
                divided from the colormap.
        """
        return self.tweak(saturation=1 / saturation)

    def __pow__(self, value: float) -> TastyMap:
        """Raises the brightness value factor to the colormap.

        Args:
            value: Brightness value factor to raise.

        Returns:
            TastyMap: A new TastyMap instance with the brightness value
                raised to the colormap.
        """
        return self.tweak(value=value)

    def __invert__(self) -> TastyMap:
        """Reverses the colormap.

        Returns:
            TastyMap: A new TastyMap instance with the reversed colormap.
        """
        return self.reverse()

    def __and__(self, tmap: TastyMap) -> TastyMap:
        """Combines two TastyMap instances.

        Args:
            tmap: Another TastyMap instance to combine with.

        Returns:
            TastyMap: A new TastyMap instance with the combined colormap.
        """
        if not isinstance(tmap, TastyMap):
            raise TypeError(
                f"Can only combine TastyMap instances; received {type(tmap)!r}."
            )
        name = self.cmap.name + "_" + tmap.cmap.name
        _cmap_array = np.concatenate([self._cmap_array, cmap_to_array(tmap.cmap)])
        cmap = LinearSegmentedColormap.from_list(name, _cmap_array, N=len(_cmap_array))
        return TastyMap(cmap)

    def __or__(self, num_colors: int) -> TastyMap:
        """Interpolates the colormap to a specified number of colors.

        Args:
            num_colors: Number of colors to interpolate to.

        Returns:
            TastyMap: A new TastyMap instance with the interpolated colormap.
        """
        return self.interpolate(num_colors)

    def __lshift__(self, name: str) -> TastyMap:
        """Renames the colormap.

        Args:
            name: New name for the colormap.

        Returns:
            TastyMap: A new TastyMap instance with the renamed colormap.
        """
        return self.rename(name)

    def __rshift__(self, name: str) -> TastyMap:
        """Registers the colormap with matplotlib.

        Args:
            name: Name of the colormap.

        Returns:
            TastyMap: A new TastyMap instance with the registered colormap.
        """
        return self.register(name)

    def __mod__(self, color_model: str) -> np.ndarray:
        """Converts the colormap to a specified color model.

        Args:
            color_model: The color model to convert to.

        Returns:
            np.ndarray: Array representation of the colormap
                in the specified color model.
        """
        return self.to(color_model)

    def __len__(self) -> int:
        """Returns the number of colors in the colormap.

        Returns:
            int: Number of colors in the colormap.
        """
        return len(self._cmap_array)

    def __eq__(self, other: TastyMap) -> bool:
        """Checks if two TastyMap instances are equal.

        Args:
            other: Another TastyMap instance to compare with.

        Returns:
            bool: True if the two TastyMap instances are equal; False otherwise.
        """
        if not isinstance(other, TastyMap):
            return False
        return np.all(self._cmap_array == other._cmap_array)

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
    bad: str | tuple | None = None,
    under: str | tuple | None = None,
    over: str | tuple | None = None,
    from_color_model: ColorModel | str | None = None,
) -> LinearSegmentedColormap:
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
            colors_or_cmap is an Iterable and not hexcodes. Defaults to None.

    Returns:
        LinearSegmentedColormap: A new colormap.
    """
    if isinstance(colors_or_cmap, str):
        tmap = TastyMap.from_str(colors_or_cmap)
    elif isinstance(colors_or_cmap, LinearSegmentedColormap):
        tmap = TastyMap(colors_or_cmap)
    elif isinstance(colors_or_cmap, ListedColormap):
        tmap = TastyMap.from_listed_colormap(colors_or_cmap)
    elif isinstance(colors_or_cmap, Iterable):
        if not isinstance(colors_or_cmap[0], str) and from_color_model is None:
            raise ValueError(
                "Please specify from_color_model to differentiate "
                "between RGB and HSV color models."
            )
        tmap = TastyMap.from_list(colors_or_cmap, color_model=from_color_model)
    else:
        raise TypeError(
            f"Expected str, LinearSegmentedColormap, ListedColormap, or Iterable; "
            f"received {type(colors_or_cmap)!r}."
        )

    if bad or under or over:
        tmap = tmap.set(bad=bad, under=under, over=over)

    if num_colors:
        tmap = tmap.interpolate(num_colors)

    if reverse:
        tmap = tmap.reverse()

    if name:
        tmap.cmap.name = name
        tmap.register()

    return tmap
