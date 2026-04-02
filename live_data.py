import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def download_stock_data():

    # Get today's date
    today = datetime.today()

    # Get past 2 years of data
    start_date = today - timedelta(days=730)

    # Download latest data
    data = yf.download("AAPL", start=start_date, end=today)

    data = data[['Close']]

    data.reset_index(inplace=True)

    data.columns = ['Date', 'Passengers']

    data.to_csv("data/raw/airline-passengers.csv", index=False)

    print("Live data updated till:", today.date())


if __name__ == "__main__":
    download_stock_data()