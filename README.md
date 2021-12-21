The deep-plats library provides functionalities for analyzing timeseries in a piecewise linear fashion with deep neural networks.

&nbsp;

Requirements
==================
- torch
- tqdm

&nbsp;

Installation
============

```bash
pip install deepplats
```

Both a `requirements.dev.txt` as well as a `requirements.dev.yml` are provided for reproducing the development environment
either through pip or conda.


&nbsp;

Minimal example
===============

In-sample forecast.

```python
from deepplats import DeepPLF
from deepplats.utils import get_data

X, y = get_data('example1')
deepplf = DeepPLF(lags=10, horizon=1, breaks=10)
deepplf.fit(x, y, epochs=1000)
deepplf.predict(x, y) # predicts on x and lagged rolling y sequences.
```

&nbsp;

Note on non-linearity
=====================

Non-linearities are used throughout the DeepPLF model. The linearity refers to the piecewise linear regression which
forms the basis on which other non-linear model extensions are added.
