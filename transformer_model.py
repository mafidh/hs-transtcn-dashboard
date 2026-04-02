import torch
import torch.nn as nn


class TransformerModel(nn.Module):

    def __init__(self, input_window=12, d_model=32, nhead=4, num_layers=2, forecast_horizon=3):

        super(TransformerModel, self).__init__()

        self.input_projection = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_model * input_window, forecast_horizon)

    def forward(self, x):

        # x shape: (batch, sequence)
        x = x.unsqueeze(-1)

        x = self.input_projection(x)

        x = x.permute(1, 0, 2)

        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2)

        x = x.flatten(start_dim=1)

        out = self.fc(x)

        return out