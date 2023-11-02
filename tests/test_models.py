import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter

from tastymap.models import ColorModel, MatplotlibTastyBar, TastyMap


@pytest.fixture
def tmap():
    cmap = LinearSegmentedColormap.from_list("testmap", ["red", "green", "blue"])
    return TastyMap(cmap)


class TestTastyMap:
    def test_init(self, tmap):
        assert tmap.cmap.name == "testmap"
        assert len(tmap._cmap_array) == 256

    def test_from_str(self):
        tmap = TastyMap.from_str("viridis")
        assert tmap.cmap.name == "viridis"

    def test_from_list(self):
        colors = ["red", "green", "blue"]
        tmap = TastyMap.from_list(colors)
        assert tmap.cmap.name == "custom_tastymap"
        assert len(tmap._cmap_array) == 3

    def test_from_list_hsv(self):
        colors = [(0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)]
        tmap = TastyMap.from_list(colors, color_model="hsv")
        assert tmap.cmap.name == "custom_tastymap"
        assert len(tmap._cmap_array) == 3

    def test_empty_colormap(self):
        with pytest.raises(ValueError):
            TastyMap.from_list([])

    def test_invalid_color(self):
        with pytest.raises(ValueError):
            TastyMap.from_list(["red", "not_a_color"])

    def test_invalid_colormap_name(self):
        with pytest.raises(ValueError):
            TastyMap.from_str("not_a_real_colormap")

    def test_non_iterable_colors(self):
        with pytest.raises(TypeError):
            TastyMap.from_list(123)

    def test_non_linearsegmented_colormap(self):
        with pytest.raises(TypeError):
            TastyMap("not_a_colormap")

    def test_interpolate(self, tmap):
        interpolated = tmap.resize(10)
        assert len(interpolated._cmap_array) == 10

    def test_reverse(self, tmap):
        reversed_map = tmap.reverse()
        assert reversed_map._cmap_array[0].tolist() == tmap._cmap_array[-1].tolist()

    def test_to(self, tmap):
        rgba_array = tmap.to_model(ColorModel.RGBA)
        assert rgba_array.shape == (256, 4)
        rgb_array = tmap.to_model(ColorModel.RGB)
        assert rgb_array.shape == (256, 3)
        hsv_array = tmap.to_model(ColorModel.HSV)
        assert hsv_array.shape == (256, 3)
        hex_array = tmap.to_model(ColorModel.HEX)
        assert hex_array.shape == (256,)

    def test_set_bad(self, tmap):
        tmap = tmap.set_extremes(bad="black", under="black", over="black")
        assert tmap.cmap.get_bad().tolist() == [0.0, 0.0, 0.0, 1.0]
        assert tmap.cmap.get_over().tolist() == [0.0, 0.0, 0.0, 1.0]
        assert tmap.cmap.get_under().tolist() == [0.0, 0.0, 0.0, 1.0]

    def test_getitem(self, tmap):
        subset = tmap[10:20]
        assert len(subset._cmap_array) == 10

    def test_and(self, tmap):
        cmap2 = LinearSegmentedColormap.from_list("testmap2", ["yellow", "cyan"])
        tmap2 = TastyMap(cmap2)
        combined = tmap & tmap2
        assert len(combined._cmap_array) == 256 + 256

    def test_len(self, tmap):
        assert len(tmap) == 256

    def test_str(self, tmap):
        assert str(tmap) == "testmap (256 colors)"

    def test_repr(self, tmap):
        assert repr(tmap) == "TastyMap('testmap')"

    def test_add_non_tastymap(self, tmap):
        with pytest.raises(TypeError):
            tmap + "not_a_tastymap"

    def test_unsupported_color_model(self, tmap):
        with pytest.raises(ValueError):
            tmap.to_model("unsupported_model")

    def test_invalid_indices(self, tmap):
        with pytest.raises(IndexError):
            tmap[1000:2000]
        with pytest.raises(TypeError):
            tmap["not_an_index"]

    def test_invalid_color_model(self, tmap):
        with pytest.raises(ValueError):
            tmap.to_model("not_a_real_color_model")

    def test_tweak_hue(self, tmap):
        tweaked = tmap.tweak_hsv(hue=50)
        assert isinstance(tweaked, TastyMap)

    def test_tweak_saturation(self, tmap):
        tweaked = tmap.tweak_hsv(saturation=5)
        assert isinstance(tweaked, TastyMap)

    def test_tweak_value(self, tmap):
        tweaked = tmap.tweak_hsv(value=2)
        assert isinstance(tweaked, TastyMap)

    def test_tweak_all(self, tmap):
        tweaked = tmap.tweak_hsv(hue=50, saturation=5, value=2)
        assert isinstance(tweaked, TastyMap)

    def test_tweak_edge_values(self, tmap):
        tweaked_min_hue = tmap.tweak_hsv(hue=-255)
        tweaked_max_hue = tmap.tweak_hsv(hue=255)
        tweaked_min_saturation = tmap.tweak_hsv(saturation=-10)
        tweaked_max_saturation = tmap.tweak_hsv(saturation=10)
        tweaked_min_value = tmap.tweak_hsv(value=0)
        tweaked_max_value = tmap.tweak_hsv(value=3)

        assert isinstance(tweaked_min_hue, TastyMap)
        assert isinstance(tweaked_max_hue, TastyMap)
        assert isinstance(tweaked_min_saturation, TastyMap)
        assert isinstance(tweaked_max_saturation, TastyMap)
        assert isinstance(tweaked_min_value, TastyMap)
        assert isinstance(tweaked_max_value, TastyMap)

    def test_tweak_out_of_range(self, tmap):
        with pytest.raises(ValueError):
            tmap.tweak_hsv(hue=300)
        with pytest.raises(ValueError):
            tmap.tweak_hsv(saturation=20)
        with pytest.raises(ValueError):
            tmap.tweak_hsv(value=4)

    def test_empty_string(self):
        with pytest.raises(ValueError):
            TastyMap.from_str("")

    def test_non_string_non_colormap(self):
        with pytest.raises(TypeError):
            TastyMap(12345)

    def test_arithmetic_with_non_numeric(self, tmap):
        with pytest.raises(TypeError):
            tmap + "string"
        with pytest.raises(TypeError):
            tmap - "string"
        with pytest.raises(TypeError):
            tmap * "string"
        with pytest.raises(TypeError):
            tmap / "string"

    def test_from_list_empty_colors(self):
        with pytest.raises(ValueError, match="Must provide at least one color."):
            TastyMap.from_list([])

    def test_and_operator_with_non_tastymap(self):
        tmap = TastyMap.from_str("viridis")
        with pytest.raises(TypeError):
            tmap & "some_string"

    def test_invert(self):
        tmap = TastyMap.from_str("viridis")
        inverted = ~tmap
        np.testing.assert_equal(inverted._cmap_array, tmap._cmap_array[::-1])

    def test_pow(self):
        tmap = TastyMap.from_str("viridis")
        result = tmap**2
        assert result == tmap.tweak_hsv(value=2)

    def test_eq_operator_with_non_tastymap(self):
        tmap = TastyMap.from_str("viridis")
        assert not (tmap == "some_string")

    def test_or_operator(self):
        tmap = TastyMap.from_str("viridis")
        result = tmap | 10
        assert len(result) == 10

    def test_lshift_operator(self):
        tmap = TastyMap.from_str("viridis")
        result = tmap << "new_name"
        assert result.cmap.name == "new_name"

    def test_rshift_operator(self):
        tmap = TastyMap.from_str("viridis")
        result = tmap >> "new_name"
        assert result.cmap.name == "new_name"

    def test_mod_operator(self):
        tmap = TastyMap.from_str("viridis")
        result = tmap % "rgb"
        assert result.shape[1] == 3

    def test_len_tmap(self):
        tmap = TastyMap.from_str("viridis")
        assert len(tmap) == 256

    def test_str_tmap(self):
        tmap = TastyMap.from_str("viridis")
        assert str(tmap) == "viridis (256 colors)"

    def test_repr_tmap(self):
        tmap = TastyMap.from_str("viridis")
        assert repr(tmap) == "TastyMap('viridis')"

    def test_iter(self):
        tmap = TastyMap.from_str("viridis")
        for color in tmap:
            assert isinstance(color, np.ndarray)

    def test_resize_with_extremes(self):
        tmap = TastyMap.from_str("viridis")
        result = tmap.set_extremes(bad="black", under="black", over="black").resize(10)
        assert result.cmap.get_bad().tolist() == [0.0, 0.0, 0.0, 1.0]
        assert result.cmap.get_over().tolist() == [0.0, 0.0, 0.0, 1.0]
        assert result.cmap.get_under().tolist() == [0.0, 0.0, 0.0, 1.0]


class TestMatplotlibTastyBar:
    @pytest.fixture
    def tmap(self):
        cmap = LinearSegmentedColormap.from_list("testmap", ["red", "green", "blue"])
        return TastyMap(cmap)

    def test_init_provided_ticks(self, tmap):
        tmap_bar = MatplotlibTastyBar(tmap, bounds=[0, 4, 18])
        np.testing.assert_equal(tmap_bar.ticks, [0, 4, 18])
        assert isinstance(tmap_bar.norm, BoundaryNorm)
        assert tmap_bar.format is None
        assert not tmap_bar.norm.clip
        assert tmap_bar.norm.extend == "both"

    def test_init_provided_ticks_and_center(self, tmap):
        tmap_bar = MatplotlibTastyBar(tmap, bounds=[0, 4, 18], center=True)
        np.testing.assert_equal(tmap_bar.ticks, [0, 2.5, 11.5])
        assert isinstance(tmap_bar.norm, BoundaryNorm)
        assert isinstance(tmap_bar.format, FuncFormatter)
        assert not tmap_bar.norm.clip
        assert tmap_bar.norm.extend == "both"

    def test_init_not_provided_ticks(self, tmap):
        tmap_bar = MatplotlibTastyBar(tmap, bounds=slice(0, 18, 4))
        assert tmap_bar.ticks is None
        assert isinstance(tmap_bar.norm, Normalize)
        assert tmap_bar.format is None
        assert not tmap_bar.norm.clip
        assert tmap_bar.norm.vmin == 0
        assert tmap_bar.norm.vmax == 18

    def test_init_not_provided_ticks_no_step(self, tmap):
        tmap_bar = MatplotlibTastyBar(tmap, bounds=slice(0, 18))
        assert tmap_bar.ticks is None
        assert isinstance(tmap_bar.norm, Normalize)
        assert tmap_bar.format is None
        assert not tmap_bar.norm.clip
        assert tmap_bar.norm.vmin == 0
        assert tmap_bar.norm.vmax == 18

    def test_plot_settings(self, tmap):
        tmap_bar = MatplotlibTastyBar(tmap, bounds=[0, 4, 18])
        plot_settings = tmap_bar.plot_settings
        assert len(plot_settings) == 2
        assert plot_settings["cmap"] == tmap.cmap
        assert plot_settings["norm"] == tmap_bar.norm

    def test_colorbar_settings(self, tmap):
        tmap_bar = MatplotlibTastyBar(tmap, bounds=[0, 4, 18])
        colorbar_settings = tmap_bar.colorbar_settings
        assert len(colorbar_settings) == 5
        np.testing.assert_equal(colorbar_settings["ticks"], tmap_bar.ticks)
        assert colorbar_settings["format"] == tmap_bar.format
        assert colorbar_settings["norm"] == tmap_bar.norm
        assert colorbar_settings["spacing"] == "uniform"
        assert isinstance(colorbar_settings["norm"], BoundaryNorm)

    def test_add_to(self, tmap):
        fig, ax = plt.subplots()
        img = ax.imshow(np.random.rand(10, 10))
        tmap_bar = MatplotlibTastyBar(tmap, bounds=[0, 4, 18])
        tmap_bar.add_to(img)
        assert len(fig.axes) == 2
