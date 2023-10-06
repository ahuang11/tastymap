import pytest
from matplotlib.colors import LinearSegmentedColormap
from tastymap.core import TastyMap, ColorModel, cook_tmap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from tastymap.core import ColorModel, cook_tmap
from tastymap.utils import get_cmap


@pytest.fixture
def tmap():
    cmap = LinearSegmentedColormap.from_list("testcmap", ["red", "green", "blue"])
    return TastyMap(cmap)


class TestTastyMapInitialization:
    def test_init(self, tmap):
        assert tmap.cmap.name == "testcmap"
        assert len(tmap.cmap_array) == 256

    def test_from_str(self):
        tmap = TastyMap.from_str("viridis")
        assert tmap.cmap.name == "viridis"

    def test_from_list(self):
        colors = ["red", "green", "blue"]
        tmap = TastyMap.from_list(colors)
        assert tmap.cmap.name == "custom_tastymap"
        assert len(tmap.cmap_array) == 256

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
        with pytest.raises(ValueError):
            TastyMap.from_list(123)

    def test_non_linearsegmented_colormap(self):
        with pytest.raises(TypeError):
            TastyMap("not_a_colormap")


class TestTastyMapMethods:
    def test_interpolate(self, tmap):
        interpolated = tmap.interpolate(10)
        assert len(interpolated.cmap_array) == 10

    def test_reverse(self, tmap):
        reversed_map = tmap.reverse()
        assert reversed_map.cmap_array[0].tolist() == tmap.cmap_array[-1].tolist()

    def test_to(self, tmap):
        rgba_array = tmap.to(ColorModel.RGBA)
        assert rgba_array.shape == (256, 4)
        rgb_array = tmap.to(ColorModel.RGB)
        assert rgb_array.shape == (256, 3)

    def test_set_bad(self, tmap):
        tmap = tmap.set(bad="black", under="black", over="black")
        assert tmap.cmap.get_bad().tolist() == [0.0, 0.0, 0.0, 1.0]
        assert tmap.cmap.get_over().tolist() == [0.0, 0.0, 0.0, 1.0]
        assert tmap.cmap.get_under().tolist() == [0.0, 0.0, 0.0, 1.0]

    def test_getitem(self, tmap):
        subset = tmap[10:20]
        assert len(subset.cmap_array) == 10

    def test_add(self, tmap):
        cmap2 = LinearSegmentedColormap.from_list("testcmap2", ["yellow", "cyan"])
        tmap2 = TastyMap(cmap2)
        combined = tmap + tmap2
        assert len(combined.cmap_array) == 256 + 256

    def test_len(self, tmap):
        assert len(tmap) == 256

    def test_str(self, tmap):
        assert str(tmap) == "testcmap (256 colors)"

    def test_repr(self, tmap):
        assert repr(tmap) == "TastyMap('testcmap')"

    def test_add_non_tastymap(self, tmap):
        with pytest.raises(TypeError):
            tmap + "not_a_tastymap"

    def test_unsupported_color_model(self, tmap):
        with pytest.raises(ValueError):
            tmap.to("unsupported_model")

    def test_invalid_num_colors(self, tmap):
        with pytest.raises(ValueError):
            TastyMap(tmap.cmap, num_colors=-5)
        with pytest.raises(ValueError):
            TastyMap(tmap.cmap, num_colors=0)

    def test_invalid_indices(self, tmap):
        with pytest.raises(IndexError):
            subset = tmap[1000:2000]
        with pytest.raises(TypeError):
            subset = tmap["not_an_index"]

    def test_invalid_color_model(self, tmap):
        with pytest.raises(ValueError):
            array = tmap.to("not_a_real_color_model")


class TestCookCmap:
    def test_cook_from_string(self):
        tmap = cook_tmap("viridis")
        assert isinstance(tmap, TastyMap)

    def test_cook_from_string_with_num_colors(self):
        tmap = cook_tmap("viridis_n10")
        assert isinstance(tmap, TastyMap)
        assert len(tmap) == 10

    def test_cook_from_string_reversed(self):
        tmap = cook_tmap("viridis_r")
        assert isinstance(tmap, TastyMap)
        # Check if the colormap is reversed by comparing the first color
        assert cook_tmap("viridis_r")[0] == cook_tmap("viridis")[255]

    def test_cook_from_listed_colormap(self):
        cmap_input = ListedColormap(["red", "green", "blue"])
        tmap = cook_tmap(cmap_input)
        assert isinstance(tmap, TastyMap)

    def test_cook_from_list(self):
        cmap_input = ["red", "green", "blue"]
        tmap = cook_tmap(cmap_input, num_colors=28)
        assert isinstance(tmap, TastyMap)
        assert len(tmap) == 28

    def test_cook_with_color_model_rgb(self):
        tmap = cook_tmap("viridis", color_model="rgb")
        assert isinstance(tmap, np.ndarray)
        assert tmap.shape[1] == 3

    def test_cook_with_color_model_hex(self):
        tmap = cook_tmap("viridis", color_model="hex")
        assert isinstance(tmap, np.ndarray)
        assert all(isinstance(color, str) and color.startswith("#") for color in tmap)

    def test_cook_with_color_model_hsv(self):
        tmap = cook_tmap("viridis", color_model="hsv")
        assert isinstance(tmap, np.ndarray)
        assert tmap.shape[1] == 3

    def test_cook_with_enum_color_model(self):
        tmap = cook_tmap("viridis", color_model=ColorModel.RGB)
        assert isinstance(tmap, np.ndarray)
        assert tmap.shape[1] == 3

    def test_r_flag_with_reverse_true(self):
        tmap = cook_tmap("viridis_r", reverse=True)
        assert isinstance(tmap, TastyMap)
        assert np.all(tmap.to("rgba")[0] == get_cmap("viridis")(0))

    def test_r_flag_with_reverse_false(self):
        tmap = cook_tmap("viridis_r", reverse=False)
        assert isinstance(tmap, TastyMap)
        assert np.all(tmap.to("rgba")[0] == get_cmap("viridis")(256))

    def test_n_flag_with_num_colors(self):
        tmap = cook_tmap("viridis_n10", num_colors=20)
        assert isinstance(tmap, TastyMap)
        assert len(tmap) == 10

    def test_n_flag_without_num_colors(self):
        tmap = cook_tmap("viridis_n10")
        assert isinstance(tmap, TastyMap)
        assert len(tmap) == 10

    def test_register(self):
        tmap = cook_tmap("viridis", name="test")
        assert plt.get_cmap("test") == tmap.cmap
