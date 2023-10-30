import ast
from typing import Sequence

import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import param
import xarray as xr
from matplotlib.colors import hsv_to_rgb, rgb2hex

from .core import cook_tmap, pair_tbar
from .models import ColorModel, TastyBar, TastyMap
from .utils import _LOWER_COLORMAPS, cmap_to_array, get_cmap

pn.extension(notifications=True)


_REGISTERED_CMAPS = {
    cmap_name: get_cmap(cmap_name) for cmap_name in _LOWER_COLORMAPS.values()
}


class TastyView(pn.viewable.Viewer):
    reverse = param.Boolean(
        default=False,
        doc="Whether to reverse the colormap. Defaults to False.",
    )

    cmap = param.ObjectSelector(
        default="magma",
        doc="The colormap to use. Defaults to 'viridis'.",
    )

    colors = param.List(
        default=None,
        doc="List of colors to use for the colormap. Defaults to None.",
        item_type=(str, tuple),
    )

    from_color_model = param.ObjectSelector(
        default=ColorModel.RGB.value.upper(),
        objects=["RGB", "HSV"],
        doc="Color model of the input colors to determine whether to use RGB or HSV.",
    )

    num_colors = param.Integer(default=11, bounds=(2, 256), doc="Number of colors.")

    name = param.String(
        default=None, doc="Name of the custom colormap. Defaults to None."
    )

    bad = param.String(
        default=None,
        doc="Color for bad values. Defaults to None.",
    )

    under = param.String(
        default=None,
        doc="Color for underflow values. Defaults to None.",
    )

    over = param.String(
        default=None,
        doc="Color for overflow values. Defaults to None.",
    )

    uniform_spacing = param.Boolean(
        default=False, doc="Whether to use uniform spacing."
    )

    bounds = param.List(
        default=None, item_type=(float, int), doc="Bounds for the colormap."
    )

    labels = param.List(default=None, item_type=str, doc="Labels for the colormap.")

    package = param.ObjectSelector(
        default="matplotlib",
        objects=["matplotlib"],
        doc="Which package to use for plotting.",
    )

    _tmap = param.ClassSelector(class_=TastyMap, doc="The tastymap.", precedence=-1)

    _tbar = param.Parameter(doc="The tastybar image.", precedence=-1)

    _active_index = param.Integer(default=0, precedence=-1)

    def __init__(self, **params):
        self._plot = pn.pane.Matplotlib(
            sizing_mode="scale_height", tight=True, max_height=300
        )
        self._history_box = pn.FlexBox()
        self._palette_box = pn.FlexBox()
        self._tmap_html = pn.pane.HTML(sizing_mode="scale_height", max_height=75)
        super().__init__(**params)

        # colors widget
        colors_input = pn.widgets.TextInput(
            name="Color Input",
            placeholder="Enter a color, or use the picker.",
            margin=(5, 5, 5, 20),
        )
        colors_picker = pn.widgets.ColorPicker(
            name="Color Picker",
            sizing_mode="stretch_width",
            margin=(5, 30, 5, 20),
        )
        self.colors_select = pn.widgets.MultiChoice.from_param(
            self.param.colors,
            name="Colors Selected",
            placeholder="Colors selected to create colormap",
            margin=(5, 5, 5, 20),
            max_items=32,
        )
        from_color_model_select = pn.widgets.Select.from_param(
            self.param.from_color_model,
            name="Color Model",
            margin=(5, 5, 5, 20),
        )
        colors_widgets = pn.Column(
            colors_input, colors_picker, self.colors_select, from_color_model_select
        )

        colors_picker.param.watch(self._add_color, "value")
        colors_input.param.watch(self._add_color, "value")

        # cmap widgets

        self.cmap_input = pn.widgets.ColorMap(
            options=_REGISTERED_CMAPS,
            ncols=2,
            swatch_width=55,
            name="Colormap",
            margin=(5, 30, 5, 20),
            sizing_mode="stretch_width",
        )
        cmap_button = pn.widgets.Button(
            name="Import Colormap Palette",
            margin=(5, 30, 5, 20),
            sizing_mode="stretch_width",
        )
        cmap_widgets = pn.Column(self.cmap_input, cmap_button)

        self.cmap_input.param.watch(self._update_cmap, "value")
        cmap_button.on_click(self._use_cmap_palette)

        # tmap widgets

        tmap_parameters = ["reverse", "num_colors", "bad", "under", "over"]
        tmap_widgets = pn.Param(
            self,
            parameters=tmap_parameters,
            show_name=False,
            widgets={
                "reverse": {
                    "type": pn.widgets.Toggle,
                    "sizing_mode": "stretch_width",
                    "margin": (5, 20, 5, 10),
                },
            },
        )

        # tbar widgets

        tbar_parameters = ["bounds", "labels", "uniform_spacing", "package"]
        tbar_widgets = pn.Param(
            self,
            parameters=tbar_parameters,
            show_name=False,
            widgets={
                "uniform_spacing": {
                    "type": pn.widgets.Toggle,
                    "sizing_mode": "stretch_width",
                    "margin": (5, 20, 5, 10),
                },
            },
        )

        # layout widgets

        self._colors_or_cmap_tabs = pn.Tabs(
            ("Colormap", cmap_widgets),
            ("Colors", colors_widgets),
            dynamic=True,
        )
        self._colors_or_cmap_tabs.link(self, active="_active_index", bidirectional=True)

        self._widgets = pn.Column(
            self._colors_or_cmap_tabs,
            pn.layout.Divider(),
            tmap_widgets,
            pn.layout.Divider(),
            tbar_widgets,
        )

        self.param.trigger("cmap")

    # event methods

    def _update_cmap(self, event):
        if not event.new:
            return
        self.cmap = self.cmap_input.value_name

    def _use_cmap_palette(self, event):
        self._active_index = 1
        num_colors = min(self.num_colors, 16)
        palette = self._tmap.resize(num_colors).to_model("hex").tolist()
        self.colors_select.param.update(
            options=palette,
            value=palette,
        )
        self._add_to_history(palette)

    def _add_color(self, event):
        new_event = event.new
        if not new_event:
            return

        if new_event.count(",") >= 2:
            new_event = ast.literal_eval(new_event)

        options = self.colors_select.options.copy()
        value = self.colors_select.value.copy()
        try:
            self.colors_select.param.update(
                options=options + [new_event] if new_event not in options else options,
                value=value + [new_event],
            )
            self._add_to_history([new_event])
        except ValueError as exc:
            self.colors_select.param.update(
                options=options,
                value=value,
            )
            pn.state.notifications.error(str(exc))
        finally:
            if isinstance(event.obj, pn.widgets.TextInput):
                event.obj.value = ""

    def _add_to_history(self, value):
        new_history = self._history_box.objects + self._render_colors(value)
        self._history_box.objects = new_history[-10:]

    # param methods
    def _render_colors(self, colors):
        color_background_map = {}

        for color in colors:
            prefix = ""
            background_color = color
            if isinstance(color, tuple):
                if any(c > 1 for c in color):
                    color = tuple(c / 255 for c in color)
                prefix = "RGB<br>"
                if self.from_color_model == "HSV":
                    background_color = hsv_to_rgb(color)
                    prefix = "HSV<br>"
                background_color = rgb2hex(background_color)
            color_background_map[f"{prefix}{color}"] = background_color
        return [
            pn.pane.HTML(
                f"<center style='background-color: lightgrey; color: black;'>{color}</center>",
                styles={
                    "background-color": background_color,
                    "font-size": "1.25em",
                },
                height=75,
                width=75,
            )
            for color, background_color in color_background_map.items()
        ]

    @pn.depends(
        "reverse",
        "cmap",
        "colors",
        "from_color_model",
        "num_colors",
        "bad",
        "under",
        "over",
        "_active_index",
        watch=True,
    )
    def _cook_tmap(self):
        if self._active_index == 1:
            colors_or_colormap = self.colors
        else:
            colors_or_colormap = get_cmap(self.cmap)

        if not colors_or_colormap:
            return

        self._tmap = cook_tmap(
            colors_or_cmap=colors_or_colormap,
            num_colors=self.num_colors,
            reverse=self.reverse,
            bad=self.bad,
            under=self.under,
            over=self.over,
            from_color_model=self.from_color_model,
        )

        if self._active_index == 1:
            colors = self.colors
        else:
            colors = self._tmap.resize(min(self.num_colors, 32)).to_model("hex")

        self._palette_box.objects = self._render_colors(colors)
        self._tmap_html.object = self._tmap.cmap._repr_html_()

    @pn.depends("_tmap", "bounds", "labels", "uniform_spacing", watch=True)
    def _pair_tbar(self):
        fig, ax = plt.subplots(facecolor="whitesmoke")
        ds = xr.tutorial.open_dataset("air_temperature")["air"].isel(time=0)
        self._mappable = ds.plot(ax=ax, add_colorbar=False)
        if self.bounds is None:
            bounds = np.linspace(
                ds.min().round(0), ds.max().round(0), min(self.num_colors - 1, 11)
            ).tolist()
        else:
            bounds = self.bounds
        self._mappable = pair_tbar(
            self._mappable,
            self._tmap,
            bounds=bounds,
            labels=self.labels if self.labels else None,
            uniform_spacing=self.uniform_spacing,
        )
        self._plot.object = fig
        plt.close(fig)

    def __panel__(self):
        return pn.template.FastListTemplate(
            sidebar=[self._widgets],
            main=[
                pn.Row(pn.pane.HTML("<h1>⏱️</h1>"), self._history_box, align="center"),
                pn.Row(pn.pane.HTML("<h1>🎨</h1>"), self._palette_box, align="center"),
                pn.Row(pn.pane.HTML("<h1>🍭</h1>"), self._tmap_html, align="center"),
                pn.Row(pn.pane.HTML("<h1>🖼️</h1>"), self._plot, align="center"),
            ],
            sidebar_width=350,
            title="👨‍🍳 TastyKitchen",
            main_max_width="clamp(800px, 80vw, 1200px)",
        )
