"""
"""
import torch


class AutoregressiveForecasting(torch.nn.Module):
    def __init__(self, lags, horizon, **kwargs):
        super().__init__()
        final_inputs, hidden_layers = self._create_hidden_layers(lags, **kwargs)
        self.hidden_model = self._create_hidden_model(hidden_layers)
        self.output_layer = self._create_output_layer(final_inputs, horizon)

    @staticmethod
    def _create_output_layer(inputs, outputs):
        output_layer = torch.nn.Linear(inputs, outputs)
        return output_layer

    @staticmethod
    def _create_hidden_model(layers):
        model = torch.nn.Sequential(*layers)
        return model

    @staticmethod
    def _create_hidden_layers(lags, **kwargs):
        hidden_layers = []
        return lags, hidden_layers

    def forward(self, X):
        staged_X = self.hidden_model(X)
        output = self.output_layer(staged_X)
        return output


class DenseAutoregressiveForecasting(AutoregressiveForecasting):
    def __init__(
        self,
        lags,
        horizon,
        n_hidden_layers=1,
        size_hidden_layers=32,
        activation="LeakyReLU",
        dropout=0.5,
    ):

        super().__init__(
            lags=lags,
            horizon=horizon,
            n_hidden_layers=n_hidden_layers,
            size_hidden_layers=size_hidden_layers,
            activation=activation,
            dropout=dropout,
        )

    @staticmethod
    def _create_hidden_layers(
        lags, n_hidden_layers, size_hidden_layers, activation, dropout
    ):
        assert n_hidden_layers >= 1, "Number of hidden layers should be >= 1."
        assert size_hidden_layers >= lags, "Size of hidden layers should be >= lags."
        hidden_layers = []
        for i in range(n_hidden_layers):
            inputs = outputs = size_hidden_layers
            if i == 0:
                inputs = lags
            hidden_layers += [torch.nn.Linear(inputs, outputs)]
            hidden_layers += [getattr(torch.nn, activation)()]
            if dropout > 0:
                hidden_layers += [torch.nn.Dropout(dropout)]
        return size_hidden_layers, hidden_layers
