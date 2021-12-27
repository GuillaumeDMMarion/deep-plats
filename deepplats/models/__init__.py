"""Import all models.
"""
from .autoregressive import (
    AutoregressiveForecasting,
    DenseAutoregressiveForecasting,
)
from .piecewiselinear import (
    PiecewiseLinearRegression,
    PiecewiseLinearForecasting,
    RNNPiecewiseLinearForecasting,
)
from .core import DeepPLF

ARF = AutoregressiveForecasting
DenseARF = DenseAutoregressiveForecasting
PLR = PiecewiseLinearRegression
PLF = PiecewiseLinearForecasting
RnnPLF = RNNPiecewiseLinearForecasting
