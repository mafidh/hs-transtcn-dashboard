import torch
import torch.nn as nn

from models.tcn_model import TCN
from models.transformer_model import TransformerModel


class HSTransTCN(nn.Module):

    def __init__(self, input_window=12, forecast_horizon=3):

        super(HSTransTCN, self).__init__()

        self.tcn = TCN(forecast_horizon=forecast_horizon)

        self.transformer = TransformerModel(
            input_window=input_window,
            forecast_horizon=forecast_horizon
        )

        # gating layer
        self.gate = nn.Linear(forecast_horizon * 2, forecast_horizon)

    def forward(self, x):

        tcn_out = self.tcn(x)

        transformer_out = self.transformer(x)

        combined = torch.cat([tcn_out, transformer_out], dim=1)

        gate_values = torch.sigmoid(self.gate(combined))

        output = gate_values * transformer_out + (1 - gate_values) * tcn_out

        return output