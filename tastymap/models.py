from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from enum import Enum
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    BoundaryNorm,
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
    hsv_to_rgb,
    rgb2hex,
    rgb_to_hsv,
)
from matplotlib.ticker import FuncFormatter

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


class TastyMap:
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
        self.cmap: Colormap = cmap
        self._cmap_array = cmap_to_array(cmap)

    @classmethod
    def from_str(cls, string: str) -> TastyMap:
        """Creates a TastyMap instance from a string name.

        Args:
            string: Name of the colormap.

        Returns:
            TastyMap: A new TastyMap instance.
        """
        cmap = get_cmap(string)  # type: ignore
        cmap_array = cmap_to_array(cmap)

        new_name = string
        cmap = LinearSegmentedColormap.from_list(
            new_name, cmap_array, N=len(cmap_array)
        )
        return TastyMap(cmap, name=new_name)

    @classmethod
    def from_list(
        cls,
        colors: Sequence,
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
        if not isinstance(colors, (Sequence, np.ndarray)):
            raise TypeError(f"Expected Sequence; received {type(colors)!r}.")
        if colors is None or len(colors) == 0:  # type: ignore
            raise ValueError("Must provide at least one color.")
        cmap_array = cmap_to_array(colors)

        if color_model in (ColorModel.HSV, ColorModel.HSV.value):
            cmap_array = hsv_to_rgb(cmap_array)

        cmap = LinearSegmentedColormap.from_list(name, cmap_array, N=len(cmap_array))
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
        return cls.from_list(listed_colormap.colors, name=name)  # type: ignore

    def _from_list_with_extremes(self, *args, **kwargs) -> LinearSegmentedColormap:
        """Creates a TastyMap instance from a list of colors with extreme values."""
        cmap = LinearSegmentedColormap.from_list(*args, **kwargs)
        cmap.set_extremes(
            bad=self.cmap.get_bad(),  # type: ignore
            under=self.cmap.get_under(),  # type: ignore
            over=self.cmap.get_over(),  # type: ignore
        )
        return cmap

    def resize(self, num_colors: int) -> TastyMap:
        """Resizes the colormap to a specified number of colors.

        Args:
            num_colors: Number of colors to resize to.

        Returns:
            TastyMap: A new TastyMap instance with the interpolated colormap.
        """
        cmap = self._from_list_with_extremes(
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
                f"Successfully registered the colormap; "
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
        bad: str | None = None,
        under: str | None = None,
        over: str | None = None,
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
        cmap.set_extremes(bad=bad, under=under, over=over)  # type: ignore
        return TastyMap(cmap)

    def tweak_hsv(
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
        cmap = self._from_list_with_extremes(
            name or self.cmap.name, cmap_array, N=len(cmap_array)
        )
        return TastyMap(cmap)

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Iterates over the colormap.

        Yields:
            np.ndarray: An array of colors.
        """
        yield from self._cmap_array

    def __getitem__(self, indices: int | float | slice | Sequence) -> TastyMap:
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
        return self.tweak_hsv(hue=hue)

    def __sub__(self, hue: float) -> TastyMap:
        """Subtracts a hue factor to the colormap.

        Args:
            hue: Hue factor to subtract.

        Returns:
            TastyMap: A new TastyMap instance with the hue
                subtracted from the colormap.
        """
        return self.tweak_hsv(hue=-hue)

    def __mul__(self, saturation: float) -> TastyMap:
        """Multiplies a saturation factor to the colormap.

        Args:
            saturation: Saturation factor to multiply.

        Returns:
            TastyMap: A new TastyMap instance with the saturation
                multiplied to the colormap.
        """
        return self.tweak_hsv(saturation=saturation)

    def __truediv__(self, saturation: float) -> TastyMap:
        """Divides a saturation factor to the colormap.

        Args:
            saturation: Saturation factor to divide.

        Returns:
            TastyMap: A new TastyMap instance with the saturation
                divided from the colormap.
        """
        return self.tweak_hsv(saturation=1 / saturation)

    def __pow__(self, value: float) -> TastyMap:
        """Raises the brightness value factor to the colormap.

        Args:
            value: Brightness value factor to raise.

        Returns:
            TastyMap: A new TastyMap instance with the brightness value
                raised to the colormap.
        """
        return self.tweak_hsv(value=value)

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
        cmap_array = np.concatenate([self._cmap_array, cmap_to_array(tmap.cmap)])
        cmap = self._from_list_with_extremes(name, cmap_array, N=len(cmap_array))
        return TastyMap(cmap)

    def __or__(self, num_colors: int) -> TastyMap:
        """Interpolates the colormap to a specified number of colors.

        Args:
            num_colors: Number of colors to resize to.

        Returns:
            TastyMap: A new TastyMap instance with the interpolated colormap.
        """
        return self.resize(num_colors)

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
        return self.to_model(color_model)

    def __len__(self) -> int:
        """Returns the number of colors in the colormap.

        Returns:
            int: Number of colors in the colormap.
        """
        return len(self._cmap_array)

    def __eq__(self, other: Any) -> bool:
        """Checks if two TastyMap instances are equal.

        Args:
            other: Another TastyMap instance to compare with.

        Returns:
            bool: True if the two TastyMap instances are equal; False otherwise.
        """
        if not isinstance(other, TastyMap):
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
        """Returns a string representation of the TastyMap instance.

        Returns:
            str: String representation of the TastyMap instance.
        """
        return f"TastyMap({self.cmap.name!r})"


class TastyBar(ABC):
    def __init__(
        self,
        tmap: TastyMap,
        bounds: slice | Sequence[float],
        labels: list[str] | None = None,
        uniform_spacing: bool = True,
    ):
        self.tmap = tmap
        self.bounds = bounds
        self.labels = labels
        self.uniform_spacing = uniform_spacing

    @abstractmethod
    def add_to(self, plot: Any):
        """Adds a colorbar to a plot.

        Args:
            plot: A plot.
        """


class MatplotlibTastyBar(TastyBar):
    def __init__(
        self,
        tmap: TastyMap,
        bounds: slice | Sequence[float],
        labels: list[str] | None = None,
        uniform_spacing: bool = True,
        center: bool | None = None,
        extend: Literal["both", "neither", "min", "max"] = "both",
        clip: bool | None = None,
        **colorbar_kwargs: dict[str, Any],
    ):
        """Initializes a MatplotlibTastyBar instance.

        Args:
            tmap: A TastyMap instance.
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
        is_slice = isinstance(self.bounds, slice)
        if is_slice:
            vmin = self.bounds.start  # type: ignore
            vmax = self.bounds.stop  # type: ignore
            step = self.bounds.step  #  type: ignore
            if step is None:
                num_ticks = min(num_colors - 1, 11)
                ticks = np.linspace(vmin, vmax, num_ticks)
            else:
                ticks = np.arange(vmin, vmax + step, step)
        else:
            ticks = np.array(self.bounds)
            ticks.sort()
            vmin, vmax = ticks[0], ticks[-1]

        if center is None and not is_slice:
            center = False

        if clip is None:
            clip = self.extend == "neither"

        norm = None
        if center is None:
            norm = Normalize(vmin=vmin, vmax=vmax, clip=clip)
            if is_slice:
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
                lambda _, index: labels[index]  # type: ignore
                if index < len(labels)  # type: ignore
                else ""
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

    def add_to(self, plot: ScalarMappable) -> ScalarMappable:
        """Adds a colorbar to a plot.

        Args:
            plot: A matplotlib ax.
        """
        plot_settings = self.plot_settings
        plot.cmap = plot_settings["cmap"]
        plot.norm = plot_settings["norm"]
        plt.colorbar(plot, **self.colorbar_settings)
        return plot


class HoloViewsTastyBar(TastyBar):
    def __init__(
        self,
        tmap: TastyMap,
        bounds: slice | Sequence[float],
        labels: list[str] | None = None,
        uniform_spacing: bool = True,
    ):
        """Initializes a HoloViewsTastyBar instance.

        Args:
            tmap: A TastyMap instance.
            bounds: Bounds for the colorbar.
            labels: Labels for the colorbar. Defaults to None.
            uniform_spacing: Whether to use uniform spacing for the colorbar.
                Defaults to True.
        """
        super().__init__(tmap, bounds, labels, uniform_spacing)

        from bokeh import models  # type: ignore

        self._models = models
        self.palette = self.tmap.to_model("hex").tolist()

        self.factors = None
        self.major_label_overrides = None

        num_colors = len(self.tmap)
        is_slice = isinstance(self.bounds, slice)
        if is_slice:
            vmin = self.bounds.start  # type: ignore
            vmax = self.bounds.stop  # type: ignore
            step = self.bounds.step  #  type: ignore
            if step is None:
                num_ticks = min(num_colors - 1, 11)
                ticks = np.linspace(vmin, vmax, num_ticks)
            else:
                ticks = np.arange(vmin, vmax + step, step)
        else:
            ticks = np.array(self.bounds)
            num_ticks = len(ticks)
        self.ticks = ticks.tolist()

        num_labels = len(labels) if labels else 0
        if uniform_spacing:
            if labels is None:
                self.factors = [
                    f"{self.ticks[i]} - {self.ticks[i + 1]}"
                    for i in range(len(self.ticks) - 1)
                ]
            else:
                self.factors = labels
                if not num_labels == num_ticks - 1:
                    raise ValueError(
                        f"Number of labels must be one less than the number of ticks; "
                        f"received {num_labels} labels and {num_ticks} ticks."
                    )
        elif not uniform_spacing and labels:
            if not num_labels == num_ticks:
                raise ValueError(
                    f"Number of labels must be equal to the number of ticks; "
                    f"received {num_labels} labels and {num_ticks} ticks."
                )
            self.major_label_overrides = dict(zip(self.ticks, labels))

    def _hook(self, hv_plot, _):
        plot = hv_plot.handles["plot"]
        mapper = self._models.CategoricalColorMapper(
            palette=self.palette,
            factors=self.factors,
        )
        color_bar = self._models.ColorBar(color_mapper=mapper)
        plot.right[0] = color_bar

    @property
    def opts_settings(self):
        """Keyword arguments for opts."""
        opts_kwargs = dict(
            cmap=self.palette,
            color_levels=self.ticks,
            clim=(self.ticks[0], self.ticks[-1]),
            colorbar=True,
        )
        colorbar_opts = dict(ticker=self._models.FixedTicker(ticks=self.ticks))
        if self.uniform_spacing:
            opts_kwargs["hooks"] = [self._hook]
        elif self.major_label_overrides:
            colorbar_opts["major_label_overrides"] = self.major_label_overrides

        if colorbar_opts:
            opts_kwargs["colorbar_opts"] = colorbar_opts
        return opts_kwargs

    def add_to(self, plot):
        """Adds a colorbar to a plot.

        Args:
            plot: A HoloViews plot.
        """
        return plot.opts(**self.opts_settings)
