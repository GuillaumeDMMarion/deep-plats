"""Core DeepPLF model.
"""
from __future__ import annotations
from typing import Optional, Union, Sequence

from tqdm import tqdm
import torch
import numpy as np
import pandas as pd

from .autoregressive import (
    AutoregressiveForecasting,
    DenseAutoregressiveForecasting,
)
from .piecewiselinear import (
    PiecewiseLinearRegression,
    PiecewiseLinearForecasting,
    RNNPiecewiseLinearForecasting,
)


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
        forecast_resid: Union[bool, str] = False,
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
            forecast_trend, lags=lags, horizon=horizon, plr=self.plr, **plf_kwargs
        )
        self.dar = self._init_dar_model(
            forecast_resid, lags=lags, horizon=horizon, **dar_kwargs
        )

    @staticmethod
    def _init_plf_model(method, lags, horizon, plr, **kwargs):
        _ = kwargs
        if method == "simple":
            return PiecewiseLinearForecasting(horizon=horizon, plr=plr)
        elif method == "rnn":
            return RNNPiecewiseLinearForecasting(
                lags=lags, horizon=horizon, plr=plr, **kwargs
            )
        raise NotImplementedError

    @staticmethod
    def _init_dar_model(method, lags, horizon, **kwargs):
        if isinstance(method, bool) and method is False:
            return None
        elif (isinstance(method, bool) and method is True) or method == "simple":
            return AutoregressiveForecasting(lags=lags, horizon=horizon)
        elif method == "dense":
            return DenseAutoregressiveForecasting(lags=lags, horizon=horizon, **kwargs)
        raise NotImplementedError

    @staticmethod
    def _extract_kwargs(kwargs: dict, prefix: str, strip: bool = True) -> dict:
        strip_func = lambda s: s.replace(f"{prefix}_", "") if strip else s
        return {strip_func(k): v for k, v in kwargs.items() if k.startswith(prefix)}

    @staticmethod
    def _nest_kwargs(kwargs: dict, prefixes: Union[list, tuple]) -> dict:
        return dict(
            zip(prefixes, map(lambda p: DeepPLF._extract_kwargs(kwargs, p), prefixes))
        )

    @staticmethod
    def _train_model(
        X: torch.Tensor,
        y: torch.Tensor,
        model: torch.nn.Model,
        epochs: int,
        batch_size: Union[int, float],
        lam: float,
        optim: str,
        lr: float,
        loss: str,
    ) -> None:
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        if isinstance(batch_size, float):
            batch_size = int(len(X) * batch_size)
        batches = int(max(np.ceil(len(X) / batch_size), 1))
        if model.name == "PiecewiseLinearRegression":
            model(X)
        optimizer = getattr(torch.optim, optim)(model.parameters(), lr=lr)
        lossfunc = getattr(torch.nn, loss)()
        for epoch in tqdm(range(epochs), desc=model.name):
            _ = epoch
            for batch in range(batches):
                X_batch = X[batch * batch_size : (batch + 1) * batch_size]
                y_batch = y[batch * batch_size : (batch + 1) * batch_size]
                y_hat = model(X_batch)
                optimizer.zero_grad()
                loss = lossfunc(y_batch, y_hat)
                if lam:
                    for k, w in model.named_parameters():
                        if k == "breaks":
                            continue
                        elif w.dim() > 1:
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

    def _fit_plr_from_array(self, X, y, epochs, **kwargs):
        base_kwargs = {
            "batch_size": 1.0,
            "lam": 0.0,
            "optim": "Adam",
            "lr": 0.001,
            "loss": "MSELoss",
        }
        self.plr.train()
        self._train_model(
            X[:, None, None],
            y[:, None, None],
            self.plr,
            epochs=epochs,
            **{**base_kwargs, **kwargs},
        )
        self.plr.eval()

    def _fit_plf_from_array(self, X, y, epochs, **kwargs):
        base_kwargs = {
            "batch_size": 0.2,
            "lam": 0.05,
            "optim": "Adam",
            "lr": 0.05,
            "loss": "MSELoss",
        }
        self.plf.train()
        seq_length = self.lags
        # y_roll = self._roll_arr(y, seq_length)[:-1]
        target = y[seq_length:]
        trend = self._call_model(X[:, None, None], self.plr).detach().numpy().flatten()
        trend_roll = self._roll_arr(trend, seq_length)[:-1]
        # x_roll = self._roll_arr(X, seq_length)[:-1]
        # Xr = np.stack([x_roll.T, y_roll.T, trend_roll.T]).T
        Xr = np.stack([trend_roll.T]).T
        yr = target[:, None]
        self._train_model(
            Xr,
            yr,
            self.plf,
            epochs=epochs,
            **{**base_kwargs, **kwargs},
        )
        self.plf.eval()

    def _fit_dar_from_array(self, X, y, epochs, **kwargs):
        base_kwargs = {
            "batch_size": 0.2,
            "lam": 0.05,
            "optim": "Adam",
            "lr": 0.05,
            "loss": "MSELoss",
        }
        self.dar.train()
        yhat = self._call_model(X[:, None, None], self.plr).detach().numpy().flatten()
        ydiff = y - yhat
        ydiffr = self._roll_arr(ydiff[self.lags :], self.horizon)
        ylaggr = self._roll_arr(y[: -self.horizon], self.lags)
        self._train_model(
            ylaggr,
            ydiffr,
            self.dar,
            epochs=epochs,
            **{**base_kwargs, **kwargs},
        )
        self.dar.eval()

    def _fit_from_array(self, X, y, epochs, batch_size, **kwargs):
        kwargs = self._nest_kwargs(kwargs, prefixes=["plr", "plf", "dar"])
        base_kwargs = {"epochs": epochs, "batch_size": batch_size}
        base_kwargs = {k: v for k, v in base_kwargs.items() if v}
        self._fit_plr_from_array(X, y, **{**base_kwargs, **kwargs["plr"]})
        if self.forecast_trend != "simple":
            self._fit_plf_from_array(X, y, **{**base_kwargs, **kwargs["plf"]})
        if self.forecast_resid is not False:
            self._fit_dar_from_array(X, y, **{**base_kwargs, **kwargs["dar"]})

    def _predict_from_array(self, X, y, mod):
        result = 0
        if mod is None or mod == "trend":
            if self.forecast_trend == "simple":
                Xr = self._roll_arr(X, self.lags)
                result += self._call_model(Xr, self.plf)
            else:
                trend = (
                    self._call_model(X[:, None, None], self.plr)
                    .detach()
                    .numpy()
                    .flatten()
                )
                trend_roll = self._roll_arr(trend, self.lags)
                Xr = np.stack([trend_roll.T]).T
                result += self._call_model(Xr, self.plf)
        if (mod is None or mod == "resid") and self.forecast_resid is not False:
            ylaggr = self._roll_arr(y, self.lags)
            result += self._call_model(ylaggr, self.dar)
        return result

    def fit(
        self,
        Xy: Union[np.ndarray, pd.Series, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        *,
        epochs: int = 100,
        batch_size: Optional[Union[int, float]] = None,
        **kwargs,
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
