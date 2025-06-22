import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
from typing import Tuple


def main():
    TICKERS = [
        "AAPL",
        "GOOG",
        "COIN",
        "AMZN",
        "TQQQ",
    ]

    ticker_dict = {}

    for tick in TICKERS:
        print(tick)
        ticker_dict[tick] = pull_data(tick)

    

def pull_data(name: str) -> Tuple[yf.Ticker, pd.DataFrame]:
    print("="*8, f"Processing {name}", "="*8, "\n", end="")
    ticker = yf.Ticker(name)
    data = ticker.history(period="10y", interval="1d", actions=False)

    # clean NaNs in yfinance data
    if data.isna().any().any():
        print(f"NaNs found in ticker: {name}")
        data.ffill(inplace=True)
        print("Data filled")

        if data.isna().any().any():
            print("WARNING. Forward filled data still contains NaNs. Data may be corrupted")
            print("Remaining NaNs in data: ")
            print(data.isnull().sum())

            print("Deleting all NaNs")
            data.dropna(inplace=True)
    else:
        print("Default data is clean")

    return ticker, data
    

def calculate_macd():
    ...
def calculate_bollinger():
    ...
def calculate_bin_label():
    ...
def calculate_delta_p():
    ...
def segment_data():
    ...
def generate_chart():
    ...

if __name__ == "__main__":
    main()