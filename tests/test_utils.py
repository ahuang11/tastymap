import pytest
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from tastymap.utils import get_cmap, cmap_to_array, sub_match


class TestGetCmap:
    def test_valid(self):
        cmap = get_cmap("viridis")
        assert isinstance(cmap, ListedColormap)

    def test_invalid(self):
        with pytest.raises(ValueError, match="Unknown colormap 'invalid_cmap'."):
            get_cmap("invalid_cmap")

    def test_suggestion(self):
        with pytest.raises(
            ValueError,
            match="Did you mean one of these",
        ):
            get_cmap("virid")

    def test_mixed_case(self):
        cmap = get_cmap("ViRiDiS")
        assert isinstance(cmap, ListedColormap)

    def test_upper_case(self):
        cmap = get_cmap("VIRIDIS")
        assert isinstance(cmap, ListedColormap)

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Unknown colormap ''."):
            get_cmap("")


class TestCmapToArray:
    def test_from_str(self):
        arr = cmap_to_array("viridis")
        assert isinstance(arr, np.ndarray)

    def test_from_linear_segmented_colormap(self):
        cmap = LinearSegmentedColormap.from_list("testing", ["red", "green", "blue"])
        arr = cmap_to_array(cmap)
        assert isinstance(arr, np.ndarray)

    def test_from_listed_colormap(self):
        cmap = ListedColormap(["red", "green", "blue"])
        arr = cmap_to_array(cmap)
        assert isinstance(arr, np.ndarray)

    def test_from_iterable(self):
        arr = cmap_to_array(["red", "green", "blue"])
        assert isinstance(arr, np.ndarray)


class TestSubMatch:
    def test_single_match(self):
        new_string, match = sub_match(r"\d+", "hello123world", "number")
        assert new_string == "helloworld"
        assert match == "123"

    def test_no_match(self):
        new_string, match = sub_match(r"\d+", "helloworld", "number")
        assert new_string == "helloworld"
        assert match == ""

    def test_multiple_matches(self):
        with pytest.raises(
            ValueError, match="Should only contain one 'number' but found .*?"
        ):
            sub_match(r"\d+", "hello123world456", "number")

    def test_empty_string(self):
        new_string, match = sub_match(r"\d+", "", "number")
        assert new_string == ""
        assert match == ""

    def test_special_regex_chars(self):
        new_string, match = sub_match(r"\.\*", "hello.*world", "regex chars")
        assert new_string == "helloworld"
        assert match == ".*"
