"""submodels init.
"""
from ._autoregressive import AutoregressiveForecasting, DenseAutoregressiveForecasting
from ._piecewiselinear import PiecewiseLinearRegression, PiecewiseLinearForecasting

ARF = AutoregressiveForecasting
DenseARF = DenseAutoregressiveForecasting
PLR = PiecewiseLinearRegression
PLF = PiecewiseLinearForecasting
