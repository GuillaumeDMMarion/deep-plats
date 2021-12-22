"""
"""
import torch


class PiecewiseLinearRegression(torch.nn.Module):
    """
    credits to: https://stackoverflow.com/users/6922739/matt-motoki
    """

    def __init__(self, breaks):
        super().__init__()
        self.breaks = self._get_default_breaks(breaks)
        self.piecewise = torch.nn.Linear(self.breaks.size(1) + 1, 1)

    @staticmethod
    def _get_default_breaks(breaks):
        if isinstance(breaks, int):
            return torch.zeros((1, breaks))
        return torch.nn.Parameter(torch.tensor([breaks], dtype=torch.float))

    def forward(self, X):
        if not isinstance(self.breaks, torch.nn.Parameter):
            self.breaks = torch.nn.Parameter(
                torch.linspace(X.min(), X.max(), self.breaks.size(1), out=self.breaks)
            )
        piecewise = self.piecewise(
            torch.cat([X, torch.nn.ReLU()(X - self.breaks)], 2)
        )  # , 1))
        return piecewise


class PiecewiseLinearForecasting(torch.nn.Module):
    def __init__(self, horizon, plr):
        super().__init__()
        self.horizon = horizon
        self.plr = self._init_plr(plr)

    @staticmethod
    def _init_plr(plr):
        if isinstance(plr, PiecewiseLinearRegression):
            return plr
        return PiecewiseLinearRegression(breaks=plr)

    def forward(self, X):
        X = X + self.horizon
        X_unsqueezed = X.unsqueeze(-1)
        plr_result = self.plr(X_unsqueezed).squeeze(-1)[:, -self.horizon :]
        return plr_result
