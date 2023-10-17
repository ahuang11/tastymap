import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.pyplot import get_cmap

from tastymap.core import cook_tmap, pair_tbar
from tastymap.models import TastyMap


class TestCookTmap:
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
            "testmap", ["red", "green", "blue"]
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
        assert np.all(tmap.to_model("rgba")[0] == get_cmap("viridis")(0))

    def test_r_flag_with_reverse_false(self):
        tmap = cook_tmap("viridis_r", reverse=False)
        assert isinstance(tmap, TastyMap)
        assert np.all(tmap.to_model("rgba")[0] == get_cmap("viridis")(256))

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


class TestPairTbar:
    def test_pair_tbar(self):
        fig, ax = plt.subplots()
        img = ax.imshow(np.random.random((10, 10)))
        tmap = cook_tmap(["red", "green", "blue"])
        pair_tbar(
            img, tmap, bounds=[0, 1], labels=["a", "b", "c"], uniform_spacing=False
        )
        assert len(fig.axes) == 2

    def test_pair_tbar_list(self):
        fig, ax = plt.subplots()
        img = ax.imshow(np.random.random((10, 10)))
        pair_tbar(
            img,
            ["red", "green", "blue"],
            bounds=[0, 1],
            labels=["a", "b", "c"],
            uniform_spacing=False,
        )
        assert len(fig.axes) == 2
