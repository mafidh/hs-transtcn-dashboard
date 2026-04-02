import torch
import torch.nn as nn


class TCN(nn.Module):

    def __init__(self, input_size=1, num_channels=32, kernel_size=3, forecast_horizon=3):
        super(TCN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=num_channels,
            kernel_size=kernel_size,
            padding=1
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            padding=1
        )

        self.fc = nn.Linear(num_channels * 12, forecast_horizon)

    def forward(self, x):

        # x shape: (batch, sequence)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = x.flatten(start_dim=1)

        out = self.fc(x)

        return out