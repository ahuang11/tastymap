import numpy as np
import pytest
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from tastymap.utils import cmap_to_array, get_cmap, replace_match, subset_cmap


class TestGetmap:
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

    def test_whitespace(self):
        with pytest.raises(ValueError, match="Unknown colormap ' viridis '."):
            get_cmap(" viridis ")

    def test_non_string_input(self):
        with pytest.raises(AttributeError):
            get_cmap(123)


class TestSubsetmap:
    @pytest.fixture
    def basic_cmap(self):
        return LinearSegmentedColormap.from_list(
            "basic", ["red", "green", "blue", "yellow"]
        )

    def test_subset_with_integer(self, basic_cmap):
        subset = subset_cmap(basic_cmap, 1)
        assert len(cmap_to_array(subset)) == 2
        assert subset.name == "basic_i1"

    def test_subset_with_float(self, basic_cmap):
        subset = subset_cmap(basic_cmap, 1.5)
        assert len(cmap_to_array(subset)) == 2
        assert subset.name == "basic_i1.5"

    def test_subset_with_slice(self, basic_cmap):
        subset = subset_cmap(basic_cmap, slice(1, 3))
        assert len(cmap_to_array(subset)) == 2
        assert subset.name == "basic_i1:3:None"

    def test_subset_with_slice_step_only(self, basic_cmap):
        subset = subset_cmap(basic_cmap, slice(None, None, 2))
        assert len(cmap_to_array(subset)) == 128
        assert subset.name == "basic_i::2"

    def test_subset_with_iterable(self, basic_cmap):
        subset = subset_cmap(basic_cmap, [0, 2])
        assert len(cmap_to_array(subset)) == 2
        assert subset.name == "basic_i0,2"

    def test_subset_with_single_iterable(self, basic_cmap):
        subset = subset_cmap(basic_cmap, [2])
        assert len(cmap_to_array(subset)) == 2
        assert subset.name == "basic_i2,2"

    def test_custom_name(self, basic_cmap):
        subset = subset_cmap(basic_cmap, 1, "custom_name")
        assert subset.name == "custom_name_i1"

    def test_invalid_indices(self, basic_cmap):
        with pytest.raises(IndexError):
            subset_cmap(basic_cmap, 1000)

    def test_invalid_iterable(self, basic_cmap):
        with pytest.raises(IndexError):
            subset_cmap(basic_cmap, [0, 1000])


class TestmapToArray:
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


class TestReplaceMatch:
    def test_single_match(self):
        new_string, match = replace_match(r"\d+", "hello123world", "number")
        assert new_string == "helloworld"
        assert match == "123"

    def test_no_match(self):
        new_string, match = replace_match(r"\d+", "helloworld", "number")
        assert new_string == "helloworld"
        assert match == ""

    def test_multiple_matches(self):
        with pytest.raises(
            ValueError, match="Should only contain one 'number' but found .*?"
        ):
            replace_match(r"\d+", "hello123world456", "number")

    def test_empty_string(self):
        new_string, match = replace_match(r"\d+", "", "number")
        assert new_string == ""
        assert match == ""

    def test_special_regex_chars(self):
        new_string, match = replace_match(r"\.\*", "hello.*world", "regex chars")
        assert new_string == "helloworld"
        assert match == ".*"

    def test_pattern_at_start(self):
        new_string, match = replace_match(r"^\d+", "123helloworld", "number")
        assert new_string == "helloworld"
        assert match == "123"

    def test_pattern_at_end(self):
        new_string, match = replace_match(r"\d+$", "helloworld123", "number")
        assert new_string == "helloworld"
        assert match == "123"

    def test_non_string_pattern(self):
        with pytest.raises(TypeError):
            replace_match(123, "helloworld", "number")

    def test_non_string_input(self):
        with pytest.raises(TypeError):
            replace_match(r"\d+", 123, "number")
