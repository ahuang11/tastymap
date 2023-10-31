import ast
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import param
import requests
import xarray as xr
from matplotlib.colors import hsv_to_rgb, rgb2hex

from .core import cook_tmap, pair_tbar
from .models import ColorModel, TastyBar, TastyMap
from .utils import _LOWER_COLORMAPS, cmap_to_array, get_cmap

pn.extension("jsoneditor", notifications=True)
pn.Column.sizing_mode = "stretch_width"


_REGISTERED_CMAPS = {
    cmap_name: get_cmap(cmap_name) for cmap_name in _LOWER_COLORMAPS.values()
}
_COPY_JS = "navigator.clipboard.writeText(source);"


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

    num_colors = param.Integer(default=11, bounds=(3, 256), doc="Number of colors.")

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

    custom_name = param.String(
        default="custom_tastymap",
        doc="Name of the custom colormap. Defaults to 'custom_tastymap'.",
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
        self._plot = pn.pane.Matplotlib(tight=True, sizing_mode="stretch_width")
        self._reference_image = pn.pane.Image(height=500, sizing_mode="stretch_width")
        self._palette_box = pn.FlexBox(min_height=100, sizing_mode="stretch_width")
        self._tmap_html = pn.pane.HTML(height=115)
        self._mpl_code_md = pn.pane.Markdown(
            name="Matplotlib",
            height=115,
            sizing_mode="stretch_width",
            margin=(-20, 0, 0, 0),
        )
        self._tmap_code_md = pn.pane.Markdown(
            name="TastyMap",
            height=115,
            sizing_mode="stretch_width",
            margin=(-20, 0, 0, 0),
        )
        self._mpl_copy_button = pn.widgets.Button(
            name="Copy to Clipboard",
            sizing_mode="fixed",
            margin=(20, 5, 5, 5),
        )
        self._tmap_copy_button = pn.widgets.Button(
            name="Copy to Clipboard",
            sizing_mode="fixed",
            margin=(20, 5, 5, 5),
        )
        self._history_box = pn.FlexBox(height=100)
        super().__init__(**params)

        # cmap widgets

        self.cmap_input = pn.widgets.ColorMap(
            options=_REGISTERED_CMAPS,
            ncols=2,
            swatch_width=55,
            name="Colormap",
            margin=(5, 20, 5, 20),
            sizing_mode="stretch_width",
        )
        cmap_button = pn.widgets.Button(
            name="Import Colormap Palette",
            margin=(5, 20, 5, 20),
            sizing_mode="stretch_width",
        )
        self.cmap_method = pn.widgets.RadioButtonGroup(
            options=[
                "Overwrite",
                "Prepend",
                "Append",
            ],
            margin=(5, 20, 5, 20),
            sizing_mode="stretch_width",
        )
        cmap_widgets = pn.Column(self.cmap_input, self.cmap_method, cmap_button)

        self.cmap_input.param.watch(self._update_cmap, "value")
        cmap_button.on_click(self._use_cmap_palette)

        # colors widgets
        self.colors_select = pn.widgets.JSONEditor.from_param(
            self.param.colors,
            sizing_mode="stretch_width",
            margin=(5, 30, 5, 20),
            menu=False,
            search=False,
        )
        from_color_model_select = pn.widgets.Select.from_param(
            self.param.from_color_model,
            name="Color Model",
            margin=(5, 5, 5, 20),
        )
        colors_input = pn.widgets.TextAreaInput(
            placeholder="Enter colors, separated by new lines.",
            margin=(5, 5, 5, 20),
            auto_grow=True,
            max_rows=3,
        )
        colors_picker = pn.widgets.ColorPicker(
            name="Color Picker",
            sizing_mode="stretch_width",
            margin=(5, 30, 5, 20),
        )
        colors_widgets = pn.Column(
            pn.Tabs(("Text", colors_input), ("Pick", colors_picker)),
            self.colors_select,
            from_color_model_select,
        )

        colors_input.param.watch(self._add_color, "value")
        colors_picker.param.watch(self._add_color, "value")

        # reference widgets

        file_input = pn.widgets.FileInput(
            accept=".png, .jpg, .jpeg, .gif, .bmp, .svg, .tiff, .tif",
            margin=(10, 30, 5, 20),
            sizing_mode="stretch_width",
        )
        url_input = pn.widgets.TextInput(
            placeholder="Enter a URL to an image to use as reference",
            margin=(10, 30, 5, 20),
            sizing_mode="stretch_width",
        )
        reference_tabs = pn.Tabs(("URL", url_input), ("File", file_input))

        url_input.param.watch(self._add_reference, "value")
        file_input.param.watch(self._add_reference, "value")

        # tmap widgets

        tmap_parameters = [
            "reverse",
            "num_colors",
            "bad",
            "under",
            "over",
            "custom_name",
        ]
        tmap_widgets = pn.Param(
            self,
            parameters=tmap_parameters,
            show_name=False,
            widgets={
                "reverse": {
                    "type": pn.widgets.Toggle,
                    "sizing_mode": "stretch_width",
                    "margin": (5, 10, 5, 10),
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
                    "margin": (5, 10, 5, 10),
                },
            },
        )

        # layout widgets

        self._colors_or_cmap_tabs = pn.Tabs(
            ("Colormap", cmap_widgets), ("Colors", colors_widgets), dynamic=True
        )
        self._colors_or_cmap_tabs.link(self, active="_active_index", bidirectional=True)

        self._widgets = pn.Column(
            self._colors_or_cmap_tabs,
            pn.layout.Divider(),
            reference_tabs,
            pn.layout.Divider(),
            tmap_widgets,
            pn.layout.Divider(),
            tbar_widgets,
        )

        self.palette_tabs = pn.Tabs(
            pn.Column(
                pn.pane.HTML("<h2>🎨 Palette</h2>"), self._palette_box, name="Palette"
            ),
            pn.Column(
                pn.pane.HTML("<h2>⏱️ History</h2>"), self._history_box, name="History"
            ),
        )
        self.colorbar_col = pn.Tabs(
            pn.Column(
                pn.pane.HTML("<h2>🍭 Colormap</h2>"), self._tmap_html, name="Colormap"
            ),
            pn.Column(
                pn.Row(
                    pn.pane.HTML("<h2>🖥️ Matplotlib Code</h2>"), self._mpl_copy_button
                ),
                self._mpl_code_md,
                name="Matplotlib",
            ),
            pn.Column(
                pn.Row(
                    pn.pane.HTML("<h2>🖥️ TastyMap Code</h2>"), self._tmap_copy_button
                ),
                self._tmap_code_md,
                name="TastyMap",
            ),
            dynamic=True,
        )
        self.image_tabs = pn.Tabs(
            pn.Column(
                pn.pane.HTML("<h2>🖼️ Example Output</h2>"),
                self._plot,
                name="Example Output",
            ),
            pn.Column(
                pn.pane.HTML("<h2>🌇 Reference Image</h2>"),
                self._reference_image,
                name="Reference Image",
            ),
            dynamic=True,
        )
        self.param.trigger("cmap")

    # event methods

    def _update_cmap(self, event):
        if not event.new:
            return
        self.cmap = self.cmap_input.value_name

    def _use_cmap_palette(self, event):
        num_colors = min(self.num_colors, 16)
        palette = self._tmap.resize(num_colors).to_model("hex").tolist()

        value = self.colors_select.value
        if isinstance(self.colors_select.value, dict):
            value = list(value)

        if self.cmap_method == "Prepend":
            value = palette + value
        elif self.cmap_method == "Overwrite":
            value = palette[:]
        else:
            value = value + palette
        self.colors_select.param.update(
            value=value,
        )
        self._active_index = 1
        self._add_to_history(palette)

    def _add_color(self, event):
        new_event = event.new
        if not new_event:
            return

        value = self.colors_select.value
        if isinstance(value, dict):
            value = list(value)

        if "\n" in new_event:
            new_event = new_event.split("\n")

        if new_event.count(",") >= 2:
            new_event = ast.literal_eval(new_event)

        if not isinstance(new_event, list):
            new_event = [new_event]

        try:
            self.colors_select.param.update(value=value + new_event)
            self._add_to_history(new_event)
        except ValueError as exc:
            self.colors_select.param.update(value=value)
            pn.state.notifications.error(str(exc))
        finally:
            if isinstance(event.obj, pn.widgets.TextInput):
                event.obj.value = ""

    def _add_reference(self, event):
        if not event.new:
            return

        try:
            if isinstance(event.new, str):
                response = requests.get(event.new)
                response.raise_for_status()
                self._reference_image.object = response.content
                event.obj.value = ""
            else:
                self._reference_image.object = event.new
        except Exception as exc:
            pn.state.notifications.error(str(exc))
        finally:
            self.image_tabs.active = 1

    def _add_to_history(self, value):
        new_history = self._history_box.objects + self._render_colors(value)
        self._history_box.objects = new_history[-26:]

    # param methods
    def _render_colors(self, colors):
        color_background_tuples = []

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
            color_background_tuples.append((f"{prefix}{color}", background_color))
        return [
            pn.pane.HTML(
                f"<center style='background-color: lightgrey; color: black;'>{color}</center>",
                styles={
                    "background-color": background_color,
                    "font-size": "1.25em",
                },
                height=75,
                width=75,
                margin=(5, 5, 15, 5),
            )
            for color, background_color in color_background_tuples
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
            colors = self._tmap.resize(min(self.num_colors, 26)).to_model("hex")

        self._palette_box.objects = self._render_colors(colors)
        self._tmap_html.object = self._tmap.cmap._repr_html_()

        colors_string = "',\n    '".join(self._tmap.to_model("hex"))
        self._mpl_code_md.object = (
            f"```python\n"
            f"from matplotlib.colors import LinearSegmentedColormap\n"
            f"colors = [\n    '{colors_string}'\n]\n"
            f"cmap = LinearSegmentedColormap.from_list({self.custom_name!r}, colors, N={self.num_colors})\n"  # noqa: E501
            f"```\n"
        )
        self._tmap_code_md.object = (
            f"```python\n"
            f"from tastymap import cook_tmap\n"
            f"colors = [\n    '{colors_string}'\n]\n"
            f"tmap = cook_tmap(colors, name={self.custom_name!r}, num_colors={self.num_colors})\n"  # noqa: E501
            f"cmap = tmap.cmap\n"
            f"```\n"
        )
        self._mpl_copy_button.js_on_click(
            args={
                "source": self._mpl_code_md.object.strip()
                .strip("`")
                .replace("python\n", "")
            },
            code=_COPY_JS,
        )
        self._tmap_copy_button.js_on_click(
            args={
                "source": self._tmap_code_md.object.strip()
                .strip("`")
                .replace("python\n", "")
            },
            code=_COPY_JS,
        )

    @pn.depends("_tmap", "bounds", "labels", "uniform_spacing", watch=True)
    def _pair_tbar(self):
        fig, ax = plt.subplots(facecolor="whitesmoke")
        ds = xr.tutorial.open_dataset("air_temperature")["air"].isel(time=0)
        self._mappable = ds.plot(ax=ax, add_colorbar=False)
        if self.bounds is None:
            ini = ds.min().round(0)
            end = ds.max().round(0)
            if self.num_colors > 18:
                bounds = slice(ini, end)
            else:
                bounds = np.linspace(ini, end, min(self.num_colors - 1, 18)).tolist()
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
            main=[self.palette_tabs, self.colorbar_col, self.image_tabs],
            sidebar_width=350,
            title="👨‍🍳 TastyKitchen",
            main_max_width="clamp(800px, 80vw, 1150px)",
        )
