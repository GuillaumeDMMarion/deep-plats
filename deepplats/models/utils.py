"""Model helper module.
"""
from __future__ import annotations
from typing import Union
import numpy as np
import torch


class Scaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    Accepts both torch.Tensor and numpy.ndarray.
    """

    def __init__(self, astype="float32"):
        self.astype = astype
        self.fitted = False
        self.mean = None
        self.std = None

    @staticmethod
    def _coerce(
        X: Union[np.ndarray, torch.Tensor], astype: str
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(X, np.ndarray):
            return X.astype(astype).copy()
        elif isinstance(X, torch.Tensor):
            return X.type(getattr(torch, astype)).clone()

    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> Scaler:
        """Extract mean and std from training."""
        mean_kwargs = std_kwargs = {}
        if isinstance(X, torch.Tensor):
            mean_kwargs = dict(keepdim=True)
            std_kwargs = dict(unbiased=False, keepdim=True)
        self.mean = X.mean(0, **mean_kwargs)
        self.std = X.std(0, **std_kwargs) + 1e-7
        self.fitted = True
        return self

    def transform(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Transform array."""
        X = self._coerce(X, self.astype)
        X -= self.mean
        X /= self.std
        return X

    def inverse_transform(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Transform array."""
        X = self._coerce(X, self.astype)
        X *= self.std
        X += self.mean
        return X

    def fit_transform(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Fit, then transform array."""
        self.fit(X)
        return self.transform(X)


class XScaler(Scaler):
    """Scaler specific for monotonically increasing timesteps."""

    def __init__(self, astype="float32"):
        self.step = None
        super().__init__(astype=astype)

    @staticmethod
    def _extract_steps(
        X: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        X_flat = X.flatten()
        steps = np.diff(X_flat) if isinstance(X, np.ndarray) else X_flat.diff()
        return steps

    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> Scaler:
        untransformed_steps = self._extract_steps(X)
        assert (
            np.unique(untransformed_steps).size == 1
        ), "Time should be monotonically increasing."
        fit_res = super().fit(X)
        transform = super().transform(X)
        self.step = self._extract_steps(transform)[0]
        return fit_res


class FlattenLSTM(torch.nn.Module):
    """LSTM flattener"""

    def __init__(self, last_step: bool = True):
        super().__init__()
        self.last_step = last_step

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Default forward method."""
        out, (final_out, _) = X
        if self.last_step:
            return final_out[0]
        return out.flatten(1)
