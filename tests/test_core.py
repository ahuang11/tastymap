import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from tastymap.core import ColorModel, TastyMap, cook_tmap
from tastymap.utils import get_cmap


@pytest.fixture
def tmap():
    cmap = LinearSegmentedColormap.from_list("testcmap", ["red", "green", "blue"])
    return TastyMap(cmap)


class TestTastyMap:
    def test_init(self, tmap):
        assert tmap.cmap.name == "testcmap"
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
        interpolated = tmap.interpolate(10)
        assert len(interpolated._cmap_array) == 10

    def test_reverse(self, tmap):
        reversed_map = tmap.reverse()
        assert reversed_map._cmap_array[0].tolist() == tmap._cmap_array[-1].tolist()

    def test_to(self, tmap):
        rgba_array = tmap.to(ColorModel.RGBA)
        assert rgba_array.shape == (256, 4)
        rgb_array = tmap.to(ColorModel.RGB)
        assert rgb_array.shape == (256, 3)
        hsv_array = tmap.to(ColorModel.HSV)
        assert hsv_array.shape == (256, 3)
        hex_array = tmap.to(ColorModel.HEX)
        assert hex_array.shape == (256,)

    def test_set_bad(self, tmap):
        tmap = tmap.set(bad="black", under="black", over="black")
        assert tmap.cmap.get_bad().tolist() == [0.0, 0.0, 0.0, 1.0]
        assert tmap.cmap.get_over().tolist() == [0.0, 0.0, 0.0, 1.0]
        assert tmap.cmap.get_under().tolist() == [0.0, 0.0, 0.0, 1.0]

    def test_getitem(self, tmap):
        subset = tmap[10:20]
        assert len(subset._cmap_array) == 10

    def test_and(self, tmap):
        cmap2 = LinearSegmentedColormap.from_list("testcmap2", ["yellow", "cyan"])
        tmap2 = TastyMap(cmap2)
        combined = tmap & tmap2
        assert len(combined._cmap_array) == 256 + 256

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

    def test_invalid_indices(self, tmap):
        with pytest.raises(IndexError):
            tmap[1000:2000]
        with pytest.raises(TypeError):
            tmap["not_an_index"]

    def test_invalid_color_model(self, tmap):
        with pytest.raises(ValueError):
            tmap.to("not_a_real_color_model")

    def test_tweak_hue(self, tmap):
        tweaked = tmap.tweak(hue=50)
        assert isinstance(tweaked, TastyMap)

    def test_tweak_saturation(self, tmap):
        tweaked = tmap.tweak(saturation=5)
        assert isinstance(tweaked, TastyMap)

    def test_tweak_value(self, tmap):
        tweaked = tmap.tweak(value=2)
        assert isinstance(tweaked, TastyMap)

    def test_tweak_all(self, tmap):
        tweaked = tmap.tweak(hue=50, saturation=5, value=2)
        assert isinstance(tweaked, TastyMap)

    def test_tweak_edge_values(self, tmap):
        tweaked_min_hue = tmap.tweak(hue=-255)
        tweaked_max_hue = tmap.tweak(hue=255)
        tweaked_min_saturation = tmap.tweak(saturation=-10)
        tweaked_max_saturation = tmap.tweak(saturation=10)
        tweaked_min_value = tmap.tweak(value=0)
        tweaked_max_value = tmap.tweak(value=3)

        assert isinstance(tweaked_min_hue, TastyMap)
        assert isinstance(tweaked_max_hue, TastyMap)
        assert isinstance(tweaked_min_saturation, TastyMap)
        assert isinstance(tweaked_max_saturation, TastyMap)
        assert isinstance(tweaked_min_value, TastyMap)
        assert isinstance(tweaked_max_value, TastyMap)

    def test_tweak_out_of_range(self, tmap):
        with pytest.raises(ValueError):
            tmap.tweak(hue=300)
        with pytest.raises(ValueError):
            tmap.tweak(saturation=20)
        with pytest.raises(ValueError):
            tmap.tweak(value=4)

    def test_empty_string(self):
        with pytest.raises(ValueError):
            TastyMap.from_str("")

    def test_non_string_non_colormap(self):
        with pytest.raises(TypeError):
            TastyMap(12345)

    def test_tweak_edge_values(self, tmap):
        tweaked = tmap.tweak(hue=255, saturation=10, value=3)
        assert isinstance(tweaked, TastyMap)

    def test_tweak_out_of_range(self, tmap):
        with pytest.raises(ValueError):
            tmap.tweak(hue=300)
        with pytest.raises(ValueError):
            tmap.tweak(saturation=20)
        with pytest.raises(ValueError):
            tmap.tweak(value=4)

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
            result = tmap & "some_string"

    def test_cook_tmap_iterable_without_color_model(self):
        with pytest.raises(ValueError, match="Please specify from_color_model"):
            cook_tmap([(0.5, 0.5, 0.5)])

    def test_invert(self):
        tmap = TastyMap.from_str("viridis")
        inverted = ~tmap
        np.testing.assert_equal(inverted._cmap_array, tmap._cmap_array[::-1])

    def test_pow(self):
        tmap = TastyMap.from_str("viridis")
        result = tmap ** 2
        assert result == tmap.tweak(value=2)

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

    def test_len(self):
        tmap = TastyMap.from_str("viridis")
        assert len(tmap) == 256

    def test_str(self):
        tmap = TastyMap.from_str("viridis")
        assert str(tmap) == "viridis (256 colors)"

    def test_repr(self):
        tmap = TastyMap.from_str("viridis")
        assert repr(tmap) == "TastyMap('viridis')"

    def test_iter(self):
        tmap = TastyMap.from_str("viridis")
        for color in tmap:
            assert isinstance(color, np.ndarray)
    

class TestCookCmap:
    def test_cook_from_string(self):
        tmap = cook_tmap("viridis")
        assert isinstance(tmap, TastyMap)

    def test_cook_from_string_reversed(self):
        tmap = cook_tmap("viridis_r")
        assert isinstance(tmap, TastyMap)
        # Check if the colormap is reversed by comparing the first color
        assert cook_tmap("viridis_r")[0] == cook_tmap("viridis")[255]

    def test_cook_from_listed_colormap(self):
        cmap_input = ListedColormap(["red", "green", "blue"])
        tmap = cook_tmap(cmap_input)
        assert isinstance(tmap, TastyMap)

    def test_cook_from_linear_segmented_colormap(self):
        cmap_input = LinearSegmentedColormap.from_list(
            "testcmap", ["red", "green", "blue"]
        )
        tmap = cook_tmap(cmap_input)
        assert isinstance(tmap, TastyMap)

    def test_cook_from_list(self):
        cmap_input = ["red", "green", "blue"]
        tmap = cook_tmap(cmap_input, num_colors=28)
        assert isinstance(tmap, TastyMap)
        assert len(tmap) == 28

    def test_cook_from_list_no_color_model(self):
        cmap_input = [(0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)]
        with pytest.raises(ValueError):
            cook_tmap(cmap_input)

    def test_r_flag_with_reverse_true(self):
        tmap = cook_tmap("viridis_r", reverse=True)
        assert isinstance(tmap, TastyMap)
        assert np.all(tmap.to("rgba")[0] == get_cmap("viridis")(0))

    def test_r_flag_with_reverse_false(self):
        tmap = cook_tmap("viridis_r", reverse=False)
        assert isinstance(tmap, TastyMap)
        assert np.all(tmap.to("rgba")[0] == get_cmap("viridis")(256))

    def test_with_num_colors(self):
        tmap = cook_tmap("viridis", num_colors=20)
        assert isinstance(tmap, TastyMap)
        assert len(tmap) == 20

    def test_register(self):
        tmap = cook_tmap("viridis", name="test")
        assert plt.get_cmap("test") == tmap.cmap

    def test_non_iterable_input(self):
        with pytest.raises(TypeError):
            cook_tmap(12345)

    def test_bad_under_over(self):
        tmap = cook_tmap("viridis", under="red", over="blue", bad="green")
        tmap.cmap.get_under().tolist() == [1.0, 0.0, 0.0, 1.0]
        tmap.cmap.get_over().tolist() == [0.0, 0.0, 1.0, 1.0]
        tmap.cmap.get_bad().tolist() == [0.0, 1.0, 0.0, 1.0]
