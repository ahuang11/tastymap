# TastyMap

## 🎨 Color palettes for your palate 😋

Make, customize, and/or use colormaps, any way you like.

![tastykitchen](https://github.com/ahuang11/tastymap/assets/15331990/ce015064-2ffb-4da2-bb8e-4818fdd751ab)

## 📖 Quick start

Start cooking from pre-made colormaps...

```python
from tastymap import cook_tmap

tmap = cook_tmap("viridis", num_colors=12, reverse=True)
tmap
```

![viridis_12](https://github.com/ahuang11/tastymap/assets/15331990/ee9b429b-26d6-4eef-8128-a93f47a920ab)

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

![rgb_256](https://github.com/ahuang11/tastymap/assets/15331990/b0964acc-56d1-4add-b9d4-fdc925756098)

Then pair it with your plots effortlessly:

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

![example](https://github.com/ahuang11/tastymap/assets/15331990/04ab9ea7-d836-44b8-843d-2cb65eddfe63)

Or if you need suggestions, get help from AI by providing a description of what you're imagining:

```python
from tastymap import ai

tmap = ai.suggest_tmap("Pikachu")
tmap
```

![image](https://github.com/ahuang11/tastymap/assets/15331990/5a6f2bd4-4c4f-449c-9f2a-3352c956400a)

Try to craft your visual delight *interactively* with the TastyKitchen UI, hosted [here](https://huggingface.co/spaces/ahuang11/tastykitchen).

```bash
tastymap ui
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

---

[![build](https://github.com/ahuang11/tastymap/workflows/Build/badge.svg)](https://github.com/ahuang11/tastymap/actions)
[![codecov](https://codecov.io/gh/ahuang11/tastymap/branch/master/graph/badge.svg)](https://codecov.io/gh/ahuang11/tastymap)
[![PyPI version](https://badge.fury.io/py/tastymap.svg)](https://badge.fury.io/py/tastymap)

**Documentation**: <a href="https://ahuang11.github.io/tastymap/">https://ahuang11.github.io/tastymap/</a>

**Source Code**: <a href="https://github.com/ahuang11/tastymap" target="_blank">https://github.com/ahuang11/tastymap</a>
