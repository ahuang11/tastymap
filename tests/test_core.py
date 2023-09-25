import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from tastymap.utils import get_cmap
from tastymap.core import cook_cmap, ColorModel


class TestCookCmap:
    def test_cook_from_string(self):
        cmap = cook_cmap("viridis")
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_cook_from_string_with_num_colors(self):
        cmap = cook_cmap("viridis_n10")
        assert isinstance(cmap, LinearSegmentedColormap)
        assert cmap.N == 10

    def test_cook_from_string_reversed(self):
        cmap = cook_cmap("viridis_r")
        assert isinstance(cmap, LinearSegmentedColormap)
        # Check if the colormap is reversed by comparing the first color
        assert np.all(cmap(0) == get_cmap("viridis")(256))

    def test_cook_from_listed_colormap(self):
        cmap_input = ListedColormap(["red", "green", "blue"])
        cmap = cook_cmap(cmap_input)
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_cook_from_list(self):
        cmap_input = ["red", "green", "blue"]
        cmap = cook_cmap(cmap_input, num_colors=28)
        assert isinstance(cmap, LinearSegmentedColormap)
        assert cmap.N == 28

    def test_cook_with_color_model_rgb(self):
        cmap = cook_cmap("viridis", color_model="rgb")
        assert isinstance(cmap, np.ndarray)
        assert cmap.shape[1] == 3

    def test_cook_with_color_model_hex(self):
        cmap = cook_cmap("viridis", color_model="hex")
        assert isinstance(cmap, np.ndarray)
        assert all(isinstance(color, str) and color.startswith("#") for color in cmap)

    def test_cook_with_color_model_hsv(self):
        cmap = cook_cmap("viridis", color_model="hsv")
        assert isinstance(cmap, np.ndarray)
        assert cmap.shape[1] == 3

    def test_cook_with_enum_color_model(self):
        cmap = cook_cmap("viridis", color_model=ColorModel.RGB)
        assert isinstance(cmap, np.ndarray)
        assert cmap.shape[1] == 3

    def test_r_flag_with_reverse_true(self):
        cmap = cook_cmap("viridis_r", reverse=True)
        assert isinstance(cmap, LinearSegmentedColormap)
        assert np.all(cmap(0) == get_cmap("viridis")(256))

    def test_r_flag_with_reverse_false(self):
        cmap = cook_cmap("viridis_r", reverse=False)
        assert isinstance(cmap, LinearSegmentedColormap)
        assert np.all(cmap(0) == get_cmap("viridis")(256))

    def test_n_flag_with_num_colors(self):
        cmap = cook_cmap("viridis_n10", num_colors=20)
        assert isinstance(cmap, LinearSegmentedColormap)
        assert cmap.N == 10

    def test_n_flag_without_num_colors(self):
        cmap = cook_cmap("viridis_n10")
        assert isinstance(cmap, LinearSegmentedColormap)
        assert cmap.N == 10

    def test_register(self):
        cmap = cook_cmap("viridis", name="test")
        assert plt.get_cmap("test") == cmap
