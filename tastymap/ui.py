import ast
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb2hex

try:
    import panel as pn  # type: ignore[import]
    import param  # type: ignore[import]
    import requests  # type: ignore[import]
    import xarray as xr  # type: ignore[import]
except ImportError:
    raise ImportError(
        "TastyKitchen additionally requires panel, param, requests, and xarray; "
        "run `pip install 'tastymap[ui]'` to install."
    )

try:
    from .ai import suggest_tmap
except ImportError:
    suggest_tmap = None

from .core import cook_tmap, pair_tbar
from .models import ColorModel, TastyMap
from .utils import get_cmap, get_registered_cmaps

pn.extension("jsoneditor", notifications=True)
pn.Column.sizing_mode = "stretch_width"


class TastyKitchen(pn.viewable.Viewer):
    reverse = param.Boolean(
        default=False,
        doc="Whether to reverse the colormap. Defaults to False.",
        label="Reverse Colormap",
    )

    cmap = param.ObjectSelector(
        default="accent",
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
        label="Bad Color",
    )

    under = param.String(
        default=None,
        doc="Color for underflow values. Defaults to None.",
        label="Underflow Color",
    )

    over = param.String(
        default=None,
        doc="Color for overflow values. Defaults to None.",
        label="Overflow Color",
    )

    hue = param.Number(
        default=0,
        bounds=(-255, 255),
        doc="Hue factor (-255 to 255) to tweak by.",
    )

    saturation = param.Number(
        default=1,
        bounds=(-10, 10),
        doc="Saturation factor (-10 to 10) to tweak by.",
    )

    value = param.Number(
        default=1,
        bounds=(0, 3),
        doc="Brightness value factor (0, 3) to tweak by.",
    )

    custom_name = param.String(
        default="custom_tastymap",
        doc="Name of the custom colormap. Defaults to 'custom_tastymap'.",
        label="Colormap name",
    )

    uniform_spacing = param.Boolean(
        default=False,
        doc="Whether to use uniform spacing.",
        label="Uniform Spacing Between Ticks",
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

    _registered_cmaps = param.Dict(doc="Registered colormaps.", precedence=-1)

    _active_index = param.Integer(default=0, precedence=-1)

    def __init__(self, **params):
        self._plot = pn.pane.Matplotlib(tight=True, sizing_mode="stretch_width")
        self._reference_image = pn.pane.Image(height=300, sizing_mode="stretch_width")
        self._palette_box = pn.FlexBox(min_height=100, sizing_mode="stretch_width")
        self._tmap_html = pn.pane.HTML(height=115)
        self._mpl_code_md = pn.pane.Markdown(
            name="Matplotlib",
            min_height=300,
            sizing_mode="stretch_width",
            margin=(-20, 0, 0, 0),
        )
        self._tmap_code_md = pn.pane.Markdown(
            name="TastyMap",
            min_height=300,
            sizing_mode="stretch_width",
            margin=(-20, 0, 0, 0),
        )
        self._history_box = pn.FlexBox(height=100)
        super().__init__(**params)

        # cmap widgets

        cmaps = {cmap_name: get_cmap(cmap_name) for cmap_name in get_registered_cmaps()}
        cmaps = self._sort_cmaps(cmaps)
        self.cmap_input = pn.widgets.ColorMap(
            options=cmaps,
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
        colors_upload = pn.widgets.FileInput(
            accept=".rgb,.txt",
            sizing_mode="stretch_width",
            margin=(10, 30, 5, 20),
        )
        if suggest_tmap is None:
            colors_suggest = pn.widgets.TextAreaInput(
                placeholder="This feature requires the `tastymap[ai]` extra.",
                margin=(5, 5, 5, 20),
                auto_grow=True,
                max_rows=3,
                disabled=True,
            )
        else:
            colors_suggest = pn.widgets.TextAreaInput(
                placeholder="Enter a description to let AI suggest a colormap",
                margin=(5, 5, 5, 20),
                auto_grow=True,
                max_rows=3,
            )
        colors_clear = pn.widgets.Button(
            name="Clear",
            sizing_mode="stretch_width",
            margin=(5, 30, 5, 20),
        )
        colors_widgets = pn.Column(
            from_color_model_select,
            pn.Tabs(
                ("Text", colors_input),
                ("Pick", colors_picker),
                ("Upload", colors_upload),
                ("Suggest", colors_suggest),
                ("Clear", colors_clear),
            ),
            self.colors_select,
        )

        colors_input.param.watch(self._add_color, "value")
        colors_picker.param.watch(self._add_color, "value")
        colors_upload.param.watch(self._add_color, "value")
        colors_suggest.param.watch(self._add_color, "value")
        colors_clear.on_click(lambda event: setattr(self.colors_select, "value", []))

        # tmap widgets

        tmap_parameters = [
            "reverse",
            "num_colors",
            "hue",
            "saturation",
            "value",
            "bad",
            "under",
            "over",
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
                "bad": {"placeholder": "black"},
                "under": {"placeholder": "blue"},
                "over": {"placeholder": "red"},
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
                "bounds": {"placeholder": "[230, 273.15, 291, 300, 330]"},
                "labels": {
                    "placeholder": "['freezing', 'cool', 'comfortable', 'warm', 'hot']"
                },
            },
        )

        # finalize widgets

        name_input = pn.widgets.TextInput.from_param(
            self.param.custom_name,
            sizing_mode="stretch_width",
            margin=(5, 30, 5, 20),
        )
        register_button = pn.widgets.Button(
            name="Register Colormap for Session",
            sizing_mode="stretch_width",
            margin=(5, 30, 5, 20),
        )
        self.colors_download = pn.widgets.FileDownload(
            filename=f"{self.custom_name}_hexcodes.txt",
            callback=pn.bind(self._download_colors, self.param._tmap),
            sizing_mode="stretch_width",
            margin=(5, 30, 5, 20),
        )
        finalize_widgets = pn.Column(
            name_input,
            register_button,
            self.colors_download,
        )

        register_button.on_click(self._register_tmap)

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

        # layout widgets

        self._colors_or_cmap_tabs = pn.Tabs(
            ("Colormap", cmap_widgets), ("Colors", colors_widgets), dynamic=True
        )
        self._colors_or_cmap_tabs.link(self, active="_active_index", bidirectional=True)

        self._widgets = pn.Column(
            self._colors_or_cmap_tabs,
            pn.layout.Divider(),
            tmap_widgets,
            pn.layout.Divider(),
            tbar_widgets,
            pn.layout.Divider(),
            finalize_widgets,
        )

        self.palette_tabs = pn.Tabs(
            pn.Column(
                pn.pane.HTML("<h2>🎨 Palette</h2>"), self._palette_box, name="Palette"
            ),
            pn.Column(
                pn.pane.HTML("<h2>⏱️ History</h2>"), self._history_box, name="History"
            ),
        )
        self.colorbar_col = pn.Column(
            pn.pane.HTML("<h2>🍭 Colormap</h2>"), self._tmap_html, name="Colormap"
        )
        self.image_tabs = pn.Tabs(
            pn.Column(
                pn.pane.HTML("<h2>🖼️ Example Output</h2>"),
                self._plot,
                name="Example Output",
            ),
            pn.Column(
                pn.pane.HTML("<h2>🌇 Reference Image</h2>"),
                reference_tabs,
                self._reference_image,
                name="Reference Image",
            ),
            pn.Column(
                pn.Row(
                    pn.pane.HTML("<h2>🖥️ Matplotlib Code</h2>"),
                ),
                self._mpl_code_md,
                name="Matplotlib Code",
            ),
            pn.Column(
                pn.Row(
                    pn.pane.HTML("<h2>🖥️ TastyMap Code</h2>"),
                ),
                self._tmap_code_md,
                name="TastyMap Code",
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

        if self.cmap_method.value == "Prepend":
            value = palette + value
        elif self.cmap_method.value == "Overwrite":
            value = palette[:]
        else:
            value = value + palette
        self.colors_select.param.update(value=value)
        self._active_index = 1
        self._add_to_history(palette)

    def _add_color(self, event):
        new_event = event.new
        if not new_event:
            return

        if isinstance(new_event, bytes):
            new_event = new_event.decode("utf-8")
        elif "let AI" in event.obj.placeholder:
            try:
                event.obj.disabled = True
                tmap = suggest_tmap(new_event, self.num_colors)
                self.custom_name = tmap.cmap.name
                new_event = tmap.to_model("hex").tolist()
            finally:
                event.obj.disabled = False

        value = self.colors_select.value
        if isinstance(value, dict):
            value = list(value)

        if " " in new_event:
            new_event = new_event.strip()

        if "\n" in new_event:
            new_event = new_event.splitlines()

        if not isinstance(new_event, list):
            new_event = [new_event]

        processed_colors = []
        for color in new_event:
            color = color.strip().strip(",")
            if not color:
                continue
            try:
                if " " in color or color.count(",") == 2:
                    color = np.array(ast.literal_eval(",".join(color.split()))).astype(
                        float
                    )
                    if any(c > 1 for c in color):
                        color /= 255
                    color = tuple(color.round(2))
            except Exception as exc:
                pn.state.notifications.error(str(exc))
                continue
            processed_colors.append(color)
        try:
            self.colors_select.param.update(value=value + processed_colors)
            self._add_to_history(processed_colors)
        except ValueError as exc:
            if "invalid" in str(exc).lower():
                self.colors_select.param.update(value=value)
                pn.state.notifications.error(str(exc))
        finally:
            if isinstance(event.obj, pn.widgets.TextInput):
                event.obj.value = ""

        try:
            self.num_colors = len(self.colors_select.value)
        except Exception:
            pass

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

    def _register_tmap(self, event):
        self._tmap.register(name=self.custom_name)
        options = self.cmap_input.options
        options[self.custom_name] = self._tmap.cmap
        self.cmap_input.options = self._sort_cmaps(options)
        pn.state.notifications.success(
            f"Registered {self.custom_name} for this session and it can now be "
            f"accessed under the Colormap tab.",
            10000,
        )

    def _sort_cmaps(self, cmaps):
        return dict(sorted(cmaps.items(), key=lambda item: item[0]))

    def _download_colors(self, tmap):
        colors_string = "\n".join(tmap.to_model("hex"))
        buf = StringIO(colors_string)
        return buf

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
                f"<center style='background-color: lightgrey; "
                f"color: black;'>{color}</center>",
                styles={
                    "background-color": background_color,
                    "font-size": "0.75em",
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
        "hue",
        "saturation",
        "value",
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
            hue=self.hue,
            saturation=self.saturation,
            value=self.value,
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

    @pn.depends("custom_name", watch=True)
    def _update_filename(self):
        self.colors_download.filename = f"{self.custom_name}_hexcodes.txt"

    def __panel__(self):
        return pn.template.FastListTemplate(
            sidebar=[self._widgets],
            main=[self.palette_tabs, self.colorbar_col, self.image_tabs],
            sidebar_width=350,
            title="👨‍🍳 TastyKitchen",
            main_max_width="clamp(800px, 80vw, 1150px)",
        )

    def serve(self, port: int = 8888, show: bool = True, **kwargs):
        pn.serve(self.__panel__(), port=port, show=show, **kwargs)
