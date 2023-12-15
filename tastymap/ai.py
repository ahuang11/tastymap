try:
    from marvin import ai_fn, ai_model  # type: ignore
    from pydantic import BaseModel, Field  # type: ignore
except ImportError:
    raise ImportError(
        "Please install marvin and pydantic to use this module, "
        "e.g. pip install marvin pydantic"
    )

from .core import cook_tmap
from .models import TastyMap


@ai_model(max_tokens=256)
class AIPalette(BaseModel):
    colors: list[str] = Field(
        default=...,
        description="""
        A list of colors as existing matplotlib named colors or hex codes,
        like `["firebrick", "#FFFFFF", "#000000"]`. If the color is invalid,
        find the closest color to the provided color.
        """,
    )
    name: str = Field(..., description="A creative name to describe the colors.")


@ai_fn(max_tokens=256)
def _refine_description(description: str, num_colors: int) -> str:  # pragma: no cover
    """
    You are a master painter, and well versed in matplotlib colors.
    Describe in detail what you imagine when you think of
    the provided `description` in descriptive named colors.

    Then, share a variety of colors, either as valid matplotlib named colors
    or hex codes that best represent the image, so that you can use
    it to paint the image, up to `num_colors` colors.
    """


def suggest_tmap(
    description: str, num_colors: int = 5, retries: int = 3, verbose: bool = True
) -> TastyMap:
    """
    Suggest a TastyMap based on a description of the image.

    Args:
        description: A description of the image.
        num_colors: Number of colors in the colormap. Defaults to 5.
        retries: Number of retries to suggest a TastyMap. Defaults to 3.
        verbose: Whether to print the AI description. Defaults to True.

    Returns:
        TastyMap: A new TastyMap instance with the new colormap.
    """
    exceptions = []
    for _ in range(retries):
        try:
            ai_description = _refine_description(description, num_colors)
            if verbose:
                print(ai_description)
            ai_palette = AIPalette(ai_description)
            return cook_tmap(
                ["".join(color.split()) for color in ai_palette.colors],
                name=ai_palette.name,
            )
        except Exception as exception:
            exceptions.append(exception)
    else:
        raise ValueError(
            f"Attempted to suggest a TastyMap {retries} times, "
            f"but failed due to {exceptions}"
        )
