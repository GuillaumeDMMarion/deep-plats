"""Module containing torch models for piecewise linear analysis.
"""
from __future__ import annotations
from typing import Union, Sequence
import torch


class PiecewiseLinearRegression(torch.nn.Module):
    """Regression for obtaining piecewise linear sections.

    Args:
        breaks: Absolute number of breaks or relative to the length of the sequence
                Initial breaks can be provided too.

    credits to: https://stackoverflow.com/users/6922739/matt-motoki
    """

    name = "PiecewiseLinearRegression"

    def __init__(self, breaks: Union[float, int, Sequence] = 0.2):
        super().__init__()
        self.breaks = breaks
        self.piecewise = None
        if not isinstance(self.breaks, (float, int)):
            self.breaks = torch.nn.Parameter(torch.tensor([breaks], dtype=torch.float))
            self.piecewise = torch.nn.Linear(self.breaks.size(1) + 1, 1)

    @staticmethod
    def _get_default_breaks(
        X: torch.Tensor, breaks: Union[float, int]
    ) -> torch.nn.Paramter:
        n_breaks = breaks if isinstance(breaks, int) else int(X.size(0) * breaks)
        return torch.nn.Parameter(
            torch.linspace(X.min(), X.max(), n_breaks, dtype=torch.float)[None, :]
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Torch.nn forward."""
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

    def __init__(
        self, horizon: int, plr: Union[float, int, Sequence, PiecewiseLinearRegression]
    ):
        super().__init__()
        self.horizon = horizon
        self.plr = self._init_plr(plr)

    @staticmethod
    def _init_plr(
        plr: Union[float, int, Sequence, PiecewiseLinearRegression]
    ) -> PiecewiseLinearRegression:
        if isinstance(plr, PiecewiseLinearRegression):
            return plr
        return PiecewiseLinearRegression(breaks=plr)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Torch.nn forward."""
        X = X + self.horizon
        X_unsqueezed = X.unsqueeze(-1)
        plr_result = self.plr(X_unsqueezed).squeeze(-1)[:, -self.horizon :]
        return plr_result
