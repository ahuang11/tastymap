from typing import Sequence

import holoviews as hv
import matplotlib.pyplot as plt
import panel as pn
import param
import xarray as xr

from .core import cook_tmap, pair_tbar
from .models import ColorModel, TastyBar, TastyMap
from .utils import _LOWER_COLORMAPS


class TastyView(pn.viewable.Viewer):
    reverse = param.Boolean(
        default=False,
        doc="Whether to reverse the colormap. Defaults to False.",
    )

    colors = param.List(
        default=None,
        doc="List of colors to use for the colormap. Defaults to None.",
        item_type=str,
    )

    from_color_model = param.ObjectSelector(
        default=ColorModel.RGB.value.upper(),
        objects=["RGB", "HSV"],
        doc="Color model of the input colors to determine whether to use RGB or HSV.",
    )

    cmap = param.ObjectSelector(
        default="viridis",
        objects=_LOWER_COLORMAPS,
        doc="The colormap to use. Defaults to 'viridis'.",
    )

    num_colors = param.Integer(default=256, bounds=(2, 256), doc="Number of colors.")

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

    _tbar = param.ClassSelector(class_=TastyBar, doc="The tastybar.", precedence=-1)

    def __init__(self, **params):
        self._plot = pn.pane.HoloViews()
        self._tmap_html = pn.pane.HTML()
        super().__init__(**params)
        colors_input = pn.widgets.TextInput(
            name="Colors Input",
            placeholder="Enter a color here in hex, RGB, or HSV.",
            margin=(5, 5, 5, 20),
        )
        self.colors_select = pn.widgets.MultiChoice.from_param(
            self.param.colors,
            name="Colors Selected",
            placeholder="Colors selected to create colormap",
            margin=(5, 5, 5, 20),
        )
        from_color_model_select = pn.widgets.Select.from_param(
            self.param.from_color_model,
            name="Color Model",
            margin=(5, 5, 5, 20),
        )
        colors_widgets = pn.Column(
            colors_input, self.colors_select, from_color_model_select
        )
        colors_input.param.watch(self._add_color, "value")

        cmap_widgets = pn.Param(self, parameters=["cmap"], show_name=False)

        tmap_parameters = ["reverse", "num_colors", "bad", "under", "over"]
        tmap_widgets = pn.Param(
            self,
            parameters=tmap_parameters,
            show_name=False,
            widgets={
                "reverse": {"type": pn.widgets.Toggle, "sizing_mode": "stretch_width"},
            },
        )

        tbar_parameters = ["bounds", "labels", "uniform_spacing", "package"]
        tbar_widgets = pn.Param(
            self,
            parameters=tbar_parameters,
            show_name=False,
            widgets={
                "uniform_spacing": {
                    "type": pn.widgets.Toggle,
                    "sizing_mode": "stretch_width",
                },
            },
        )

        self._colors_or_cmap_tabs = pn.Tabs(
            ("Colormap", cmap_widgets),
            ("Colors", colors_widgets),
            dynamic=True,
        )
        self._widgets = pn.Column(
            pn.layout.Divider(),
            self._colors_or_cmap_tabs,
            tmap_widgets,
            pn.layout.Divider(),
            tbar_widgets,
        )

    def _add_color(self, event):
        if not event.new:
            return
        self.colors_select.param.update(
            options=self.colors_select.options + [event.new],
            value=self.colors_select.value + [event.new],
        )
        event.obj.value = ""

    @pn.depends("package", watch=True)
    def _create_plot(self):
        fig, ax = plt.subplots()
        ds = xr.tutorial.open_dataset("air_temperature")["air"].isel(time=0)
        self._mappable = ds.plot(ax=ax)
        self._plot.object = fig

    @pn.depends(
        "reverse",
        "colors",
        "from_color_model",
        "cmap",
        "num_colors",
        "bad",
        "under",
        "over",
        watch=True,
    )
    def _cook_tmap(self):
        if self._colors_or_cmap_tabs.active == 1:
            colors_or_colormap = self.colors
        else:
            colors_or_colormap = self.cmap
        self._tmap = cook_tmap(
            colors_or_cmap=colors_or_colormap,
            num_colors=self.num_colors,
            reverse=self.reverse,
            bad=self.bad,
            under=self.under,
            over=self.over,
            from_color_model=self.from_color_model,
        )
        self._tmap_html.object = self._tmap._repr_html_()

    # @pn.depends("_tmap", "bounds", "labels", "uniform_spacing", watch=True)
    # def _pair_tbar(self):
    #     self._tbar = pair_tbar(
    #         self._mappable,
    #         self._tmap,
    #         self.bounds,
    #         self.labels,
    #         self.uniform_spacing,
    #     )

    def __panel__(self):
        return pn.template.FastListTemplate(
            sidebar=[self._widgets],
            main=[self._plot, self._tmap_html],
            sidebar_width=350,
        )
