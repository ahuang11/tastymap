# Usage

## Creating a `TastyMap`

Creating a `TastyMap` is as simple as passing a list of named colors:

```python
from tastymap import cook_tmap

tmap = cook_tmap(["red", "green", "blue"])
```

You may also use hex colors, RGB tuples, and HSV tuples:

```python
from tastymap import cook_tmap

tmap = cook_tmap(["#ff0000", "#00ff00", "#0000ff"])
```

Or even registered colormaps with customizations:

```python
from tastymap import cook_tmap

tmap = cook_tmap("RdBu_r", num_colors=11)
```

## Using a `TastyMap`

After you have a `TastyMap`, you can use it in your plots by passing a name:

```python
from tastymap import cook_tmap
import matplotlib.pyplot as plt
import numpy as np

tmap = cook_tmap("RdBu_r", num_colors=11, name="RdBu_r_11")

data = np.random.rand(10, 10)
plt.imshow(data, cmap='RdBu_r_11')
plt.colorbar()
```

Or if you don't want to name it, you can access the underlying `LinearSegmentedColormap` object:

```python
from tastymap import cook_tmap
import matplotlib.pyplot as plt
import numpy as np

tmap = cook_tmap("RdBu_r", num_colors=11)
cmap = tmap.cmap

data = np.random.rand(10, 10)
plt.imshow(data, cmap=cmap)
plt.colorbar()
```

If you want better control over the bounds, ticks, and labels of the resulting colorbar, you can use the `pair_tbar` function:

```python
from tastymap import cook_tmap, pair_tbar
import matplotlib.pyplot as plt
import numpy as np

tmap = cook_tmap("RdBu_r", num_colors=11)

data = np.random.rand(10, 10)
img = plt.imshow(data)
pair_tbar(img, tmap, bounds=[0, 0.18, 1], labels=["zero", ".18", "one"], uniform_spacing=True)
```

You can also use it with HoloViews / hvPlot:

```python
from tastymap import cook_tmap, pair_tbar
import holoviews as hv
import numpy as np

hv.extension('bokeh')

tmap = cook_tmap("RdBu_r", num_colors=4)

data = np.random.rand(10, 10)
img = hv.Image(data)
pair_tbar(img, tmap, bounds=[0, 0.2, 0.5, 0.7, 1], uniform_spacing=True)
```

## Customizing a `TastyMap`

You can customize a `TastyMap` by passing in some or all of these keyword arguments:

```python
from tastymap import cook_tmap

cook_tmap(
    colors_or_cmap=["red", "green", "blue"],
    num_colors=18,
    reverse=True,
    hue=1.28,
    saturation=0.5,
    value=1.18,
    bad="gray",
    under="red",
    over="blue",
    name="custom_rgb_18",
)
```

Or, you can use the methods on the `TastyMap` object:

```python
from tastymap import cook_tmap

tmap = (
    cook_tmap(["red", "green", "blue"])
    .resize(18)
    .reverse.tweak_hsv(1.28, 0.5, 1.18)
    .set_extremes("gray", "red", "blue")
    .register("custom_rgb_18")
)
```

## Getting the color palette

You can get the color palette as a list of hex colors:

```python
from tastymap import cook_tmap

tmap = cook_tmap(["red", "green", "blue"])
colors = tmap.to_model("hex")
```

Or a list of RGB and HSV tuples:

```python
from tastymap import cook_tmap

tmap = cook_tmap(["red", "green", "blue"])
colors = tmap.to_model("rgb")
```

## Joining two `TastyMap`s

You can combine two `TastyMap`s with `&`:

```python
from tastymap import cook_tmap

tmap1 = cook_tmap(["red", "green", "blue"])
tmap2 = cook_tmap(["yellow", "cyan", "magenta"])

tmap = tmap1 & tmap2
```

## Tweaking with operators

You can tweak the hue by adding or subtracting:

```python
from tastymap import cook_tmap

tmap = cook_tmap(["red", "green", "blue"])
tmap + 10 - 5
```

You can tweak the saturation by multiplying or dividing:

```python
from tastymap import cook_tmap

tmap = cook_tmap(["red", "green", "blue"])
tmap * 0.5 / 0.25
```

You can tweak the brightness value by exponentiating:

```python
from tastymap import cook_tmap

tmap = cook_tmap(["red", "green", "blue"])
tmap ** 2
```

You can reverse the order of the colors with `~`:

```python
from tastymap import cook_tmap

tmap = cook_tmap(["red", "green", "blue"])
~tmap
```

You can rename a `TastyMap` with `<<` (remember as input):

```python
from tastymap import cook_tmap

tmap = cook_tmap(["red", "green", "blue"])
tmap << "rgb"
```

You can register a `TastyMap` with `>>` (remember as output):

```python
from tastymap import cook_tmap

tmap = cook_tmap(["red", "green", "blue"])
tmap >> "rgb"
```

## Using the TastyKitchen UI

You can use the TastyKitchen UI to craft your `TastyMap` interactively:

```bash
tastymap ui
```

Be sure to first install `tastymap` with the `ui` extra:

```bash
pip install tastymap[ui]
```
