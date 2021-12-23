"""Core DeepPLF model.
"""
from __future__ import annotations
from typing import Optional, Union, Sequence
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd

from ._submodels import *


class DeepPLF:
    """Deep piecewise linear forecaster.

    Args:
        lags:
        horizon:
        breaks:
        lam:
        forecast_trend:
        forecast_resid:
        dar_kwargs:
        plf_kwargs:
    """

    def __init__(
        self,
        lags: int,
        horizon: int,
        breaks: Union[float, int, Sequence] = 0.1,
        lam: float = 0.1,
        forecast_trend: str = "simple",
        forecast_resid: str = "dense",
        dar_kwargs: Optional[dict] = None,
        plf_kwargs: Optional[dict] = None,
    ):
        self.lags = lags
        self.horizon = horizon
        self.breaks = breaks
        self.lam = lam
        self.forecast_trend = forecast_trend
        self.forecast_resid = forecast_resid
        dar_kwargs = {} if dar_kwargs is None else dar_kwargs
        plf_kwargs = {} if plf_kwargs is None else plf_kwargs
        self.plr = PiecewiseLinearRegression(breaks)
        self.plf = self._init_plf_model(
            forecast_trend, horizon=horizon, plr=self.plr, **plf_kwargs
        )
        self.dar = self._init_dar_model(
            forecast_resid, lags=lags, horizon=horizon, **dar_kwargs
        )

    @staticmethod
    def _init_plf_model(method, horizon, plr, **kwargs):
        _ = kwargs
        if method == "simple":
            return PiecewiseLinearForecasting(horizon=horizon, plr=plr)

    @staticmethod
    def _init_dar_model(method, lags, horizon, **kwargs):
        if (isinstance(method, bool) and method is True) or method == "simple":
            return AutoregressiveForecasting(lags=lags, horizon=horizon)
        elif method == "dense":
            return DenseAutoregressiveForecasting(lags=lags, horizon=horizon, **kwargs)

    @staticmethod
    def _train_model(
        X,
        y,
        model,
        epochs,
        batch_size=None,
        lam=0,
        optim="Adam",
        lr=0.1,
        loss="MSELoss",
    ):
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        batch_size = len(X) if batch_size is None else batch_size
        batches = int(max(np.ceil(len(X) / batch_size), 1))
        if model.name == "PiecewiseLinearRegression":
            model(X)
        optimizer = getattr(torch.optim, optim)(model.parameters(), lr=lr)
        lossfunc = getattr(torch.nn, loss)()
        for epoch in tqdm(range(epochs), desc=model.name):
            _ = epoch
            for batch in range(batches):
                # print(
                # f'epoch: {epoch+1}/{epochs}',
                # f'batch: {batch+1}/{batches}',
                # f'batch_size: {batch_size}')
                X_batch = X[batch * batch_size : (batch + 1) * batch_size]
                y_batch = y[batch * batch_size : (batch + 1) * batch_size]
                y_hat = model(X_batch)
                optimizer.zero_grad()
                loss = lossfunc(y_batch, y_hat)
                if lam:
                    for w in model.parameters():
                        if w.dim() > 1:
                            loss = loss + lam * w.norm(1)
                loss.backward()
                optimizer.step()

    @staticmethod
    def _call_model(X, model):
        X = torch.tensor(X, dtype=torch.float)
        return model(X)

    @staticmethod
    def _roll_arr(arr, window):
        shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
        strides = arr.strides + (arr.strides[-1],)
        result = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
        return result.copy()

    def _fit_from_array(self, X, y, epochs, batch_size, **kwargs):
        self.plr.train(True)
        self._train_model(
            X[:, None, None],
            y[:, None, None],
            self.plr,
            epochs=epochs,
            batch_size=batch_size,
            lam=self.lam,
            **kwargs
        )
        self.plr.eval()
        if self.forecast_trend != "simple":
            raise NotImplementedError
            Xr = self._roll_arr(X, self.horizon)
            yr = self._roll_arr(y, self.horizon)
            self._train_model(
                Xr,
                yr,
                self.plf,
                epochs=epochs,
                batch_size=batch_size,
                lam=self.lam,
                **kwargs
            )
        if self.forecast_resid is not False:
            yhat = (
                self._call_model(X[:, None, None], self.plr).detach().numpy().flatten()
            )
            ydiff = y - yhat
            ydiffr = self._roll_arr(ydiff[self.lags :], self.horizon)
            ylaggr = self._roll_arr(y[: -self.horizon], self.lags)
            self._train_model(
                ylaggr,
                ydiffr,
                self.dar,
                epochs=epochs,
                batch_size=batch_size,
                lam=self.lam,
                **kwargs
            )

    def _predict_from_array(self, X, y, mod):
        Xr = self._roll_arr(X, self.lags)
        result = 0
        ylaggr = self._roll_arr(y, self.lags)
        if mod is None or mod == "trend":
            self.plf.eval()
            result += self._call_model(Xr, self.plf)
        if (mod is None or mod == "resid") and self.forecast_resid is not False:
            self.dar.eval()
            result += self._call_model(ylaggr, self.dar)
        return result

    def fit(
        self,
        Xy: Union[np.ndarray, pd.Series, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        *,
        epochs: int,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> DeepPLF:
        """Fit on data."""
        if isinstance(Xy, pd.DataFrame):
            X = Xy.index
            y = Xy.iloc[:, 1].values
        elif isinstance(Xy, pd.Series):
            X = Xy.index
            y = Xy.values
        elif isinstance(Xy, np.ndarray) and y is not None:
            X = Xy
        else:
            raise NotImplementedError
        return self._fit_from_array(
            X, y, epochs=epochs, batch_size=batch_size, **kwargs
        )

    def predict(
        self,
        Xy: Union[np.ndarray, pd.Series, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        mod: Optional[str] = None,
        keep_type: bool = True,
    ) -> np.ndarray:
        """Predict from data."""
        if isinstance(Xy, pd.DataFrame):
            to_type = pd.DataFrame
            X = Xy.index.values
            y = Xy.iloc[:, 0].values
        elif isinstance(Xy, pd.Series):
            to_type = pd.Series
            X = Xy.index.values
            y = Xy.values
        elif isinstance(Xy, np.ndarray) and y is not None:
            to_type = lambda arr: arr
            X = Xy
        else:
            raise NotImplementedError
        pred = self._predict_from_array(X, y, mod=mod)
        if keep_type:
            return to_type(pred.detach().numpy().flatten())
        return pred

    def forecast(self):
        """Forecast n steps."""
        raise NotImplementedError
