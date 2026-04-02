import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):

    def __init__(self, file_path, input_window=12, forecast_horizon=3):

        df = pd.read_csv(file_path)

        # Convert Date column
        self.dates = pd.to_datetime(df["Date"])

        values = df["Passengers"].values

        self.X = []
        self.y = []
        self.date_labels = []

        for i in range(len(values) - input_window - forecast_horizon):

            x = values[i:i+input_window]
            y = values[i+input_window:i+input_window+forecast_horizon]

            self.X.append(x)
            self.y.append(y)

            # store corresponding future dates
            self.date_labels.append(
                self.dates[i+input_window:i+input_window+forecast_horizon]
            )

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.date_labels[idx]