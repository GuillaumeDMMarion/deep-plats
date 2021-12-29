"""Module containing torch models for piecewise linear analysis.
"""
from __future__ import annotations
from typing import Union, Sequence, Generator

import torch

from .utils import FlattenLSTM, XScaler


class PiecewiseLinearRegression(torch.nn.Module):
    """Regression for obtaining piecewise linear sections.

    Args:
        breaks: Either number of breaks, in absolute terms or relative to the sequence length,
                or an initial sequence of breaks.
        scale: Whether to scale X by default or not. If set to False,
               for best results scale X prior to fitting.

    credits to: https://stackoverflow.com/users/6922739/matt-motoki
    """

    name = "PiecewiseLinearRegression"

    def __init__(self, breaks: Union[float, int, Sequence] = 0.1, scale: bool = True):

        super().__init__()
        self.breaks = breaks
        self.scale = scale
        self.scaler = XScaler() if self.scale else None
        self.piecewise = None
        if not isinstance(self.breaks, (float, int)):
            self.breaks = torch.nn.Parameter(
                torch.tensor([breaks], dtype=torch.float32)
            )
            self.piecewise = torch.nn.Linear(self.breaks.size(1) + 1, 1)

    @staticmethod
    def _get_default_breaks(
        X: torch.Tensor, breaks: Union[float, int]
    ) -> torch.nn.Paramter:
        n_breaks = breaks if isinstance(breaks, int) else int(X.size(0) * breaks)
        return torch.nn.Parameter(
            torch.linspace(X.min(), X.max(), n_breaks, dtype=torch.float32)[None, :]
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Torch.nn forward."""
        if self.scale:
            if not self.scaler.fitted:
                self.scaler.fit(X)
            X = self.scaler.transform(X).clone()
        if not isinstance(self.breaks, torch.nn.Parameter):
            self.breaks = self._get_default_breaks(X, self.breaks)
            self.piecewise = torch.nn.Linear(self.breaks.size(1) + 1, 1)
        piecewise = self.piecewise(
            torch.cat([X, torch.nn.ReLU()(X - self.breaks)], 2)
        )  # , 1))
        return piecewise


class PiecewiseLinearForecasting(torch.nn.Module):
    """Base forecasting (on) piecewise linear sections.

    Args:
        horizon: Future number of steps to forecast.
        plr: A breaks paramter value or the actual regression model.
    """

    name = "PiecewiseLinearForecasting"

    def __init__(
        self,
        horizon: int = 1,
        plr: Union[float, int, Sequence, PiecewiseLinearRegression] = 0.2,
    ):
        super().__init__()
        self.horizon = horizon
        with torch.no_grad():
            self.plr = self._init_plr(plr)

    @staticmethod
    def _init_plr(
        plr: Union[float, int, Sequence, PiecewiseLinearRegression]
    ) -> PiecewiseLinearRegression:
        if isinstance(plr, PiecewiseLinearRegression):
            return plr
        return PiecewiseLinearRegression(breaks=plr)

    @staticmethod
    def _plr_extrapolation(
        X: torch.Tensor, plr: PiecewiseLinearRegression, horizon: int, step: float
    ) -> torch.Tensor:
        """Default extrapolation method"""
        X = X + (horizon * step)
        X_unsqueezed = X.unsqueeze(-1)
        plr_extrapolation = plr(X_unsqueezed).squeeze(-1)[:, -horizon:]
        return plr_extrapolation

    def named_parameters(self, prefix="", recurse: bool = True) -> Generator:
        """Generate all but 'plr' parameters.
        Note: Effectively overrides parameters method too.
        """
        return (
            item
            for item in super().named_parameters(prefix=prefix, recurse=recurse)
            if not item[0].startswith("plr.")
        )

    def extrapolate(self, X: torch.Tensor, horizon=None) -> torch.Tensor:
        """Extrapolation method"""
        horizon = horizon if horizon else self.horizon
        step = self.plr.scaler.step if self.plr.scale else X.flatten().diff()[0]
        return self._plr_extrapolation(X, plr=self.plr, horizon=horizon, step=step)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Torch.nn forward."""
        result = self.extrapolate(X)
        return result


class RNNPiecewiseLinearForecasting(PiecewiseLinearForecasting):
    """RNN-based forecasting (on) piecewise linear sections.

    Args:
        lags: Length of the input sequence.
        horizon: Future number of steps to forecast.
        plr: A breaks paramter value or the actual regression model.
        model: Specific RNN model to be used. One of 'RNN', 'GRU' or 'LSTM'.
    """

    def __init__(
        self,
        lags: int = 10,
        horizon: int = 1,
        plr: Union[float, int, Sequence, PiecewiseLinearRegression] = 0.2,
        model: str = "LSTM",
        n_features: int = 1,
        hidden_size: int = 2,
        num_layers: int = 1,
        last_step: bool = True,
    ):
        super().__init__(horizon=horizon, plr=plr)
        self.model = model
        self.seq_length = lags
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.last_step = last_step
        self.linear_input_size = self.hidden_size
        if not self.last_step:
            self.hidden_size *= self.seq_length
        self.rnn = getattr(torch.nn, self.model)(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.flatten = FlattenLSTM(last_step=self.last_step)
        self.out = torch.nn.Linear(self.linear_input_size, self.horizon)  # 1

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Torch.nn forward"""
        if self.plr.scale:
            X = self.plr.scaler.transform(X).clone()
        res = self.rnn(X)
        res = self.flatten(res)
        out = self.out(res)
        return out
