# TastyMap

## 🎨 Color palettes for your palate 😋

Make, customize, and/or use colormaps, any way you like.

## 📖 Example

Start cooking from pre-made colormaps...

```python
from tastymap import cook_tmap

tmap = cook_tmap("viridis", num_colors=12, reverse=True)
tmap
```

Or start from scratch!

```python
from tastymap import cook_tmap

tmap = cook_tmap(
    ["red", "green", "blue"],
    num_colors=256,
    reverse=True,
    name="rgb",
)
tmap
```

Then pair them with your plots effortlessly:

```python
import numpy as np
from matplotlib import pyplot as plt
from tastymap import cook_tmap, pair_tbar

fig, ax = plt.subplots()
img = ax.imshow(np.random.random((10, 10)))
tmap = cook_tmap(["red", "green", "blue"], num_colors=256)
pair_tbar(
    img,
    tmap,
    bounds=[0, 0.01, 0.5, 1],
    labels=["zero", "tiny", "half", "one"],
    uniform_spacing=True,
)
```

Check out the [docs](https://ahuang11.github.io/tastymap) for more recipes!

## 📦 Installation

To get started on your culinary color journey, install `tastymap` with:

```bash
pip install tastymap
```

To get access to TastyKitchen UI, install `tastymap` with:

```bash
pip install tastymap[ui]
```

Then start the UI with:

```bash
tastymap ui
```

---

[![build](https://github.com/ahuang11/tastymap/workflows/Build/badge.svg)](https://github.com/ahuang11/tastymap/actions)
[![codecov](https://codecov.io/gh/ahuang11/tastymap/branch/master/graph/badge.svg)](https://codecov.io/gh/ahuang11/tastymap)
[![PyPI version](https://badge.fury.io/py/tastymap.svg)](https://badge.fury.io/py/tastymap)

**Documentation**: <a href="https://ahuang11.github.io/tastymap/">https://ahuang11.github.io/tastymap/</a>

**Source Code**: <a href="https://github.com/ahuang11/tastymap" target="_blank">https://github.com/ahuang11/tastymap</a>
