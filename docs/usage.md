# Usage

## Creating a `TastyMap`

Creating a `TastyMap` is as simple as passing a list of named colors:

```python
from tastymap import cook_tmap

tmap = cook_tmap(["red", "green", "blue"])
```

![image](https://github.com/ahuang11/tastymap/assets/15331990/0497deb3-0585-49cb-a0cc-2bf339a029ba)

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

![image](https://github.com/ahuang11/tastymap/assets/15331990/69439d81-24e7-44ef-ac07-04b654eec437)

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

![matplotlib](https://github.com/ahuang11/tastymap/assets/15331990/5101f271-73e2-483e-a3df-e7d0f586da07)

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

![pair_mpl](https://github.com/ahuang11/tastymap/assets/15331990/1cd95779-98f0-407d-a0dc-23360a40be89)

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

<img width="356" alt="pair hv" src="https://github.com/ahuang11/tastymap/assets/15331990/b9d751e0-a7df-4378-8348-455cbaed34be">

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

![image](https://github.com/ahuang11/tastymap/assets/15331990/aa9e696a-9a4f-4a51-8e78-d20052b24045)

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

You can get the color palette as an array of hex colors:

```python
from tastymap import cook_tmap

tmap = cook_tmap(["red", "green", "blue"])
colors = tmap.to_model("hex")
```

```
array(['#ff0000', '#008000', '#0000ff'], dtype='<U7')
```

Or a list of RGB and HSV tuples:

```python
from tastymap import cook_tmap

tmap = cook_tmap(["red", "green", "blue"])
colors = tmap.to_model("rgb")
```

```
array([[1.        , 0.        , 0.        ],
       [0.        , 0.50196078, 0.        ],
       [0.        , 0.        , 1.        ]])
```

## Joining two `TastyMap`s

You can combine two `TastyMap`s with `&`:

```python
from tastymap import cook_tmap

tmap1 = cook_tmap(["red", "green", "blue"])
tmap2 = cook_tmap(["yellow", "cyan", "magenta"])

tmap = tmap1 & tmap2
```

![image](https://github.com/ahuang11/tastymap/assets/15331990/eeb0cf08-4b9b-4f57-95d3-13d6b3bdd447)

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

You can rename a `TastyMap` with `<<` (input):

```python
from tastymap import cook_tmap

tmap = cook_tmap(["red", "green", "blue"])
tmap << "rgb"
```

You can register a `TastyMap` with `>>` (output):

```python
from tastymap import cook_tmap

tmap = cook_tmap(["red", "green", "blue"])
tmap >> "rgb"
```

## Suggesting based on a description

You can have AI suggest a `TastyMap` based on a description:

```python
from tastymap import ai

tmap = ai.suggest_tmap("Pikachu")
```

![image](https://github.com/ahuang11/tastymap/assets/15331990/5a6f2bd4-4c4f-449c-9f2a-3352c956400a)

## Using the TastyKitchen UI

You can use the TastyKitchen UI to craft your `TastyMap` interactively:

```bash
tastymap ui
```

![tastykitchen](https://github.com/ahuang11/tastymap/assets/15331990/ce015064-2ffb-4da2-bb8e-4818fdd751ab)

Be sure to first install `tastymap` with the `ui` extra:

```bash
pip install tastymap[ui]
```
