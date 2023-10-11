from __future__ import annotations
from abc import abstractmethod, ABC

from collections.abc import Generator, Sequence
from enum import Enum
from typing import Any, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import (
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    hsv_to_rgb,
    rgb2hex,
    rgb_to_hsv,
    BoundaryNorm,
    Normalize,
)
from matplotlib.ticker import FuncFormatter, Formatter

from .utils import cmap_to_array, get_cmap, subset_cmap


class ColorModel(Enum):
    """Enumeration for different color models.

    Attributes:
        rgba: Red, green, blue, alpha.
        rgb: Red, green, blue.
        hsv: Hue, saturation, value.
        hex: Hexcode.
    """

    RGBA = "rgba"
    RGB = "rgb"
    HSV = "hsv"
    HEX = "hex"


class TastyColorMap:
    """A class to represent and manipulate colormaps in a tasty manner.

    Attributes:
        cmap: The colormap object.
        cmap_array: RGB array representation of the colormap.
    """

    def __init__(
        self,
        cmap: Colormap,
        name: str | None = None,
    ):
        """Initializes a TastyColorMap instance.

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
        self.cmap: Colormap = cmap
        self._cmap_array = cmap_to_array(cmap)

    @classmethod
    def from_str(cls, string: str) -> TastyColorMap:
        """Creates a TastyColorMap instance from a string name.

        Args:
            string: Name of the colormap.

        Returns:
            TastyColorMap: A new TastyColorMap instance.
        """
        cmap = get_cmap(string)  # type: ignore
        cmap_array = cmap_to_array(cmap)

        new_name = string
        cmap = LinearSegmentedColormap.from_list(
            new_name, cmap_array, N=len(cmap_array)
        )
        return TastyColorMap(cmap, name=new_name)

    @classmethod
    def from_list(
        cls,
        colors: Sequence,
        name: str = "custom_tastymap",
        color_model: ColorModel | str = ColorModel.RGBA,
    ) -> TastyColorMap:
        """Creates a TastyColorMap instance from a list of colors.

        Args:
            colors: List of colors.
            name: Name of the colormap. Defaults to "custom_tastymap".

        Returns:
            TastyColorMap: A new TastyColorMap instance.
        """
        if not isinstance(colors, Sequence):
            raise TypeError(f"Expected Sequence; received {type(colors)!r}.")
        if colors is None or len(colors) == 0:  # type: ignore
            raise ValueError("Must provide at least one color.")
        cmap_array = cmap_to_array(colors)

        if color_model in (ColorModel.HSV, ColorModel.HSV.value):
            cmap_array = hsv_to_rgb(cmap_array)

        cmap = LinearSegmentedColormap.from_list(name, cmap_array, N=len(cmap_array))
        return TastyColorMap(cmap)

    @classmethod
    def from_listed_colormap(
        cls,
        listed_colormap: ListedColormap,
        name: str = "custom_tastymap",
    ) -> TastyColorMap:
        """Creates a TastyColorMap instance from a ListedColormap.

        Args:
            listed_colormap: The colormap to be used.
            name: Name of the colormap. Defaults to "custom_tastymap".

        Returns:
            TastyColorMap: A new TastyColorMap instance.
        """
        return cls.from_list(listed_colormap.colors, name=name)  # type: ignore

    def resize(self, num_colors: int) -> TastyColorMap:
        """Resizes the colormap to a specified number of colors.

        Args:
            num_colors: Number of colors to resize to.

        Returns:
            TastyColorMap: A new TastyColorMap instance with the interpolated colormap.
        """
        cmap = LinearSegmentedColormap.from_list(
            self.cmap.name, self._cmap_array, N=num_colors
        )  # TODO: reset extremes with helper func
        return TastyColorMap(cmap)

    def register(self, name: str | None = None, echo: bool = True) -> TastyColorMap:
        """Registers the colormap with matplotlib.

        Returns:
            TastyColorMap: A new TastyColorMap instance with the registered colormap.
        """
        tmap = self.rename(name) if name else self
        colormaps.register(self.cmap, name=tmap.cmap.name, force=True)
        if echo:
            print(
                f"Sucessfully registered the colormap; "
                f"to use, set `cmap={tmap.cmap.name!r}` in your plot."
            )
        return tmap

    def rename(self, name: str) -> TastyColorMap:
        """Renames the colormap.

        Args:
            name (str): New name for the colormap.

        Returns:
            TastyColorMap: A new TastyColorMap instance with the renamed colormap.
        """
        return TastyColorMap(self.cmap, name=name)

    def reverse(self) -> TastyColorMap:
        """Reverses the colormap.

        Returns:
            TastyColorMap: A new TastyColorMap instance with the reversed colormap.
        """
        return self[::-1]

    def to_model(self, color_model: ColorModel | str) -> np.ndarray:
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
            return np.apply_along_axis(
                rgb2hex, 1, self._cmap_array[:, :3]  # type: ignore
            )

    def set_extremes(
        self,
        bad: str | tuple | None = None,
        under: str | tuple | None = None,
        over: str | tuple | None = None,
    ) -> TastyColorMap:
        """Sets the colors for bad, underflow, and overflow values.

        Args:
            bad: Color for bad values. Defaults to None.
            under: Color for underflow values. Defaults to None.
            over: Color for overflow values. Defaults to None.

        Returns:
            TastyColorMap: A new TastyColorMap instance with the updated colormap.
        """
        cmap = self.cmap.copy()
        cmap.set_extremes(bad=bad, under=under, over=over)  # type: ignore
        return TastyColorMap(cmap)

    def tweak_hsv(
        self,
        hue: float | None = None,
        saturation: float | None = None,
        value: float | None = None,
        name: str | None = None,
    ) -> TastyColorMap:
        """Tweaks the hue, saturation, and value of the colormap.

        Args:
            hue: Hue factor (-255 to 255) to tweak by.
            saturation: Saturation factor (-10 to 10) to tweak by.
            value: Brightness value factor (0, 3) to tweak by.
            name: Name of the new colormap.

        Returns:
            TastyColorMap: A new TastyColorMap instance with the tweaked colormap.
        """
        cmap_array = self._cmap_array.copy()
        cmap_array[:, :3] = rgb_to_hsv(cmap_array[:, :3])
        if hue is not None:
            hue /= 255
            if abs(hue) > 1:
                raise ValueError("Hue must be between -255 and 255 (non-inclusive).")
            cmap_array[:, 0] = (cmap_array[:, 0] + hue) % 1
        if saturation is not None:
            if abs(saturation) > 10:
                raise ValueError("Saturation must be between -10 and 10.")
            cmap_array[:, 1] = cmap_array[:, 1] * saturation
        if value is not None:
            if value < 0 or value > 3:
                raise ValueError("Value must be between 0 and 3.")
            cmap_array[:, 2] = cmap_array[:, 2] * value
        cmap_array[:, :3] = hsv_to_rgb(np.clip(cmap_array[:, :3], 0, 1))
        cmap = LinearSegmentedColormap.from_list(
            name or self.cmap.name, cmap_array, N=len(cmap_array)
        )
        return TastyColorMap(cmap)

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Iterates over the colormap.

        Yields:
            np.ndarray: An array of colors.
        """
        yield from self._cmap_array

    def __getitem__(self, indices: int | float | slice | Sequence) -> TastyColorMap:
        """Gets a subset of the colormap.

        Args:
            indices: Indices to subset the colormap.

        Returns:
            TastyColorMap: A new TastyColorMap instance with the subset colormap.
        """
        cmap = subset_cmap(self.cmap, indices)
        return TastyColorMap(cmap)

    def _repr_html_(self) -> str:
        """Returns an HTML representation of the colormap.

        Returns:
            str: HTML representation of the colormap.
        """
        return self.cmap._repr_html_()

    def __add__(self, hue: float) -> TastyColorMap:
        """Adds a hue factor to the colormap.

        Args:
            hue: Hue factor to add.

        Returns:
            TastyColorMap: A new TastyColorMap instance with the hue
                added to the colormap.
        """
        return self.tweak_hsv(hue=hue)

    def __sub__(self, hue: float) -> TastyColorMap:
        """Subtracts a hue factor to the colormap.

        Args:
            hue: Hue factor to subtract.

        Returns:
            TastyColorMap: A new TastyColorMap instance with the hue
                subtracted from the colormap.
        """
        return self.tweak_hsv(hue=-hue)

    def __mul__(self, saturation: float) -> TastyColorMap:
        """Multiplies a saturation factor to the colormap.

        Args:
            saturation: Saturation factor to multiply.

        Returns:
            TastyColorMap: A new TastyColorMap instance with the saturation
                multiplied to the colormap.
        """
        return self.tweak_hsv(saturation=saturation)

    def __truediv__(self, saturation: float) -> TastyColorMap:
        """Divides a saturation factor to the colormap.

        Args:
            saturation: Saturation factor to divide.

        Returns:
            TastyColorMap: A new TastyColorMap instance with the saturation
                divided from the colormap.
        """
        return self.tweak_hsv(saturation=1 / saturation)

    def __pow__(self, value: float) -> TastyColorMap:
        """Raises the brightness value factor to the colormap.

        Args:
            value: Brightness value factor to raise.

        Returns:
            TastyColorMap: A new TastyColorMap instance with the brightness value
                raised to the colormap.
        """
        return self.tweak_hsv(value=value)

    def __invert__(self) -> TastyColorMap:
        """Reverses the colormap.

        Returns:
            TastyColorMap: A new TastyColorMap instance with the reversed colormap.
        """
        return self.reverse()

    def __and__(self, tmap: TastyColorMap) -> TastyColorMap:
        """Combines two TastyColorMap instances.

        Args:
            tmap: Another TastyColorMap instance to combine with.

        Returns:
            TastyColorMap: A new TastyColorMap instance with the combined colormap.
        """
        if not isinstance(tmap, TastyColorMap):
            raise TypeError(
                f"Can only combine TastyColorMap instances; received {type(tmap)!r}."
            )
        name = self.cmap.name + "_" + tmap.cmap.name
        cmap_array = np.concatenate([self._cmap_array, cmap_to_array(tmap.cmap)])
        cmap = LinearSegmentedColormap.from_list(name, cmap_array, N=len(cmap_array))
        return TastyColorMap(cmap)

    def __or__(self, num_colors: int) -> TastyColorMap:
        """Interpolates the colormap to a specified number of colors.

        Args:
            num_colors: Number of colors to resize to.

        Returns:
            TastyColorMap: A new TastyColorMap instance with the interpolated colormap.
        """
        return self.resize(num_colors)

    def __lshift__(self, name: str) -> TastyColorMap:
        """Renames the colormap.

        Args:
            name: New name for the colormap.

        Returns:
            TastyColorMap: A new TastyColorMap instance with the renamed colormap.
        """
        return self.rename(name)

    def __rshift__(self, name: str) -> TastyColorMap:
        """Registers the colormap with matplotlib.

        Args:
            name: Name of the colormap.

        Returns:
            TastyColorMap: A new TastyColorMap instance with the registered colormap.
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
        return self.to_model(color_model)

    def __len__(self) -> int:
        """Returns the number of colors in the colormap.

        Returns:
            int: Number of colors in the colormap.
        """
        return len(self._cmap_array)

    def __eq__(self, other: Any) -> bool:
        """Checks if two TastyColorMap instances are equal.

        Args:
            other: Another TastyColorMap instance to compare with.

        Returns:
            bool: True if the two TastyColorMap instances are equal; False otherwise.
        """
        if not isinstance(other, TastyColorMap):
            return False
        cmap_array = other._cmap_array
        return bool(np.all(self._cmap_array == cmap_array))

    def __str__(self) -> str:
        """Returns the name of the colormap.

        Returns:
            str: Name of the colormap.
        """
        return f"{self.cmap.name} ({len(self)} colors)"

    def __repr__(self) -> str:
        """Returns a string representation of the TastyColorMap instance.

        Returns:
            str: String representation of the TastyColorMap instance.
        """
        return f"TastyColorMap({self.cmap.name!r})"


class TastyColorBar(ABC):
    def __init__(
        self,
        tmap: TastyColorMap,
        bounds: tuple[float, float] | Sequence[float],
        labels: list[str] | None = None,
        uniform_spacing: bool = True,
    ):
        self.tmap = tmap
        self.bounds = bounds
        self.labels = labels
        self.uniform_spacing = uniform_spacing

    @abstractmethod
    def add_to(self, plot: Any) -> None:
        """Adds a colorbar to a plot.

        Args:
            plot: A plot.
        """


class MatplotlibTastyColorBar(TastyColorBar):
    def __init__(
        self,
        tmap: TastyColorMap,
        bounds: slice[float, float, float] | Sequence[float],
        labels: list[str] | None = None,
        uniform_spacing: bool = True,
        center: bool | None = None,
        extend: Literal["both", "neither", "min", "max"] = "both",
        clip: bool | None = None,
        **colorbar_kwargs: dict[str, Any],
    ):
        """Initializes a MatplotlibTastyColorBar instance.

        Args:
            tmap: A TastyColorMap instance.
            bounds: Bounds for the colorbar.
            labels: Labels for the colorbar. Defaults to None.
            uniform_spacing: Whether to use uniform spacing for the colorbar.
                Defaults to True.
            center: Whether to center the colorbar. Defaults to None.
            extend: Whether to extend the colorbar. Defaults to "both".
            clip: Whether to clip the colorbar. Defaults to None.
            **colorbar_kwargs: Keyword arguments for the colorbar.
        """
        super().__init__(tmap, bounds, labels, uniform_spacing)
        self.extend = extend
        self.clip = clip
        self.colorbar_kwargs = colorbar_kwargs
        self.spacing = "uniform" if uniform_spacing else "proportional"

        num_colors = len(self.tmap)
        provided_ticks = not isinstance(self.bounds, slice)
        if provided_ticks:
            ticks = np.array(self.bounds)
            ticks.sort()
            vmin, vmax = ticks[0], ticks[-1]
        else:
            vmin = self.bounds.start
            vmax = self.bounds.stop
            step = self.bounds.step
            if step is None:
                num_ticks = min(num_colors - 1, 11)
                ticks = np.linspace(vmin, vmax, num_ticks)
            else:
                ticks = np.arange(vmin, vmax + step, step)

        if center is None and provided_ticks:
            center = False

        if clip is None:
            clip = self.extend == "neither"

        norm = None
        if center is None:
            norm = Normalize(vmin=vmin, vmax=vmax, clip=clip)
            if not provided_ticks:
                ticks = None  # let matplotlib decide
        elif center:
            norm_bins = ticks + 0.5
            norm_bins = np.insert(norm_bins, 0, norm_bins[0] - 1)
            norm = BoundaryNorm(norm_bins, num_colors, clip=clip, extend=self.extend)
            if labels is None:
                labels = ticks.copy()
            ticks = norm_bins[:-1] + np.diff(norm_bins) / 2
        else:
            norm = BoundaryNorm(ticks, num_colors, clip=clip, extend=self.extend)

        format = None
        if labels is not None:
            format = FuncFormatter(
                lambda _, index: labels[index] if index < len(labels) else ""
            )

        self.norm = norm
        self.ticks = ticks
        self.format = format

    @property
    def plot_settings(self):
        """Keyword arguments for the plot."""
        return dict(cmap=self.tmap.cmap, norm=self.norm)

    @property
    def colorbar_settings(self):
        """Keyword arguments for the colorbar."""
        colorbar_kwargs = dict(
            cmap=self.tmap.cmap,
            norm=self.norm,
            ticks=self.ticks,
            format=self.format,
            spacing=self.spacing,
        )
        colorbar_kwargs.update(self.colorbar_kwargs)
        return colorbar_kwargs

    def add_to(self, plot: plt.Axes) -> None:
        """Adds a colorbar to a plot.

        Args:
            plot: A matplotlib ax.
        """
        plot.set(**self.plot_settings)
        plt.colorbar(plot, **self.colorbar_settings)
        return plot
