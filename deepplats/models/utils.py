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

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> Scaler:
        """Extract mean and std from training."""
        mean_kwargs = std_kwargs = {}
        if isinstance(X, torch.Tensor):
            mean_kwargs = dict(keepdim=True)
            std_kwargs = dict(unbiased=False, keepdim=True)
        self.mean = X.mean(0, **mean_kwargs)
        self.std = X.std(0, **std_kwargs)
        return self

    def transform(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Transform array."""
        X = X.copy()
        X -= self.mean
        X /= self.std + +1e-7
        return X

    def fit_transform(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Fit, then transform array."""
        self.fit(X)
        return self.transform(X)


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
