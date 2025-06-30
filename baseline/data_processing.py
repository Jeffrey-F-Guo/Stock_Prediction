import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
from typing import Tuple
import pandas_ta as ta
import os
import shutil
from typing import List
from datetime import datetime, date
from sklearn.preprocessing import StandardScaler

def normalize_data(data_dict):
    scaler = StandardScaler()
    train_data = pd.DataFrame(scaler.fit_transform(data_dict["train"]), columns=["Open", "Close", "High", "Low", "Volume", "Percent"])
    dev_data = pd.DataFrame(scaler.transform(data_dict["dev"]), columns=["Open", "Close", "High", "Low", "Volume", "Percent"])
    test_data = pd.DataFrame(scaler.transform(data_dict["test"]), columns=["Open", "Close", "High", "Low", "Volume", "Percent"])

    return {
        "train": (train_data),
        "dev": (dev_data),
        "test": (test_data),
    }

def data_split(train_date:str, dev_date:str, data:pd.DataFrame):
    '''given two dataframes inputs and target containing all ticker OHLCV data. Shapes will be features x tickers x time length'''
    datetime_train = datetime.strptime(train_date, "%Y-%m-%d")
    datetime_dev = datetime.strptime(dev_date, "%Y-%m-%d")

    if datetime_dev <= datetime_train:
        print("Error: end of training period must be before end of dev period")
        return -1 # replace with something better

    train_data =  data.loc[:train_date]
    dev_data = data.loc[train_date:dev_date]
    test_data = data.loc[dev_date:]

    return {
        "train": (train_data),
        "dev": (dev_data),
        "test": (test_data),
    }

def get_and_process_data(tickers:List, save_dir:str, enable_charts: bool = False):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    # final storage for data. will be 1 x (total number of 90-day sequences across all stocks)
    # each entry will be a ((90x5), 1) a 90-day sequence of ochlv data and the target prediction for the 91st day--the next day
    # tickers will be intermingled which is fine because we're treating them as general stock data instead of ticker-specific data
    # dataloader will be able to shuffle and randomly pull 90-day chunks from this
    all_train_inputs = []
    all_train_targets = []  
    train_dir = os.path.join(save_dir, "train")

    all_dev_inputs = []
    all_dev_targets = []  
    dev_dir = os.path.join(save_dir, "dev")

    all_test_inputs = []
    all_test_targets = []  
    test_dir = os.path.join(save_dir, "test")

    ticker_dict = {}
    window_size = 90
    train_date = "2024-01-01"
    dev_date = "2025-01-01"

    for tick in tickers:
        print(tick)
        ticker_obj, ticker_df = pull_data(tick)
        # calculate_macd(ticker_df)
        # calculate_bollinger(ticker_df)
        # calculate_bin_label(ticker_df)
        calculate_delta_p(ticker_df)
        # ticker_dict[tick] = (ticker_obj, ticker_df)

        # data is split by date, so have to split first before segmenting
        split_data_dict = data_split(train_date, dev_date, ticker_df)
        normalized_data_dict = normalize_data(split_data_dict)

        train_segments = segment_data(normalized_data_dict["train"])
        dev_segments = segment_data(normalized_data_dict["dev"])
        test_segments = segment_data(normalized_data_dict["test"])

        
        for idx, data in enumerate(train_segments):
            input, target = extract_targets(data)
            if (enable_charts):
                generate_chart(input, window_size*idx,tick, train_dir, window_size)
            all_train_inputs.append(input.to_numpy())
            all_train_targets.append(target)

        for idx, data in enumerate(dev_segments):
            input, target = extract_targets(data)
            if (enable_charts):
                generate_chart(input, window_size*idx,tick, dev_dir, window_size)
            all_dev_inputs.append(input.to_numpy())
            all_dev_targets.append(target)

        for idx, data in enumerate(test_segments):
            input, target = extract_targets(data)
            if (enable_charts):
                generate_chart(input, window_size*idx,tick, test_dir, window_size)
            all_test_inputs.append(input.to_numpy())
            all_test_targets.append(target)

    all_train_inputs = np.array(all_train_inputs)
    all_train_targets = np.array(all_train_targets)

    all_dev_inputs = np.array(all_dev_inputs)
    all_dev_targets = np.array(all_dev_targets)

    all_test_inputs = np.array(all_test_inputs)
    all_test_targets = np.array(all_test_targets) 

    all_data_dict = {
        "train": {"inputs":all_train_inputs, "targets":all_train_targets},
        "dev": {"inputs":all_dev_inputs, "targets":all_dev_targets},
        "test": {"inputs":all_test_inputs, "targets":all_test_targets},
    }

    return all_data_dict


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
    

def calculate_macd(df: pd.DataFrame):
    df.ta.macd(close="Close", fast=13, slow=26, signal=9, append=True)

def calculate_bollinger(df: pd.DataFrame):
    df.ta.bbands(close='Close', length=20, std=2.0, append=True)

def calculate_bin_label(df: pd.DataFrame):
    """Calculate the binary up/down label for the next day"""
    # input dataframe will have Open High Low Close Volume
    if "Close" not in df.columns:
        print("Data does not have close. Cannot calculate")
        return
    if "Close_delta" not in df.columns:
        df["Close_delta"] = df["Close"].diff()
    if "U/D" not in df.columns:
        df["U/D"] = np.where(df["Close_delta"] > 0, 1, 0)


def calculate_delta_p(df: pd.DataFrame):
    """Calculate the binary up/down label for the next day"""
    # input dataframe will have Open High Low Close Volume
    if "Close" not in df.columns:
        print("Data does not have close. Cannot calculate")
        return
    if "Percent" not in df.columns:
        df["Percent"] = df["Close"].pct_change() * 100
        # shift labels back by one day to simulate prediction. Each day's condition aim to predict the up/down status
        # of the next day
        df["Percent"] = df["Percent"].shift(-1)
        df.dropna(inplace=True)

def extract_targets(df:pd.DataFrame) -> Tuple[pd.DataFrame, np.float32]:
    '''extract the targets of a single ticker. returned outputs will be concated to a main input and target dataframe'''
    columns = ["Open", "Close", "High", "Low", "Volume", "Percent"]
    if any(col not in df.columns for col in columns):
        print("input dataframe does not match expected format")
        return None, None
    # precondition that percent has been aggregated and shifted properly already
    inputs = df.drop(columns=["Percent"])
    target = df["Percent"].iloc[-1]

    return (inputs, target.astype(np.float32))

def segment_data(df: pd.DataFrame, window_size: int=90, stride:int = 10)->List:
    # 3 options for sampling items:
    # 1. separate non-overlapping chunks of size seq_len
    # 2. sliding window stride 1
    # 3. sliding window stride k where k != 1

    # each array index contains a 90-day sequence
    ticker_segments = [df.iloc[idx:idx+window_size] for idx in range(0, len(df)-window_size, stride)]

    # add the most recent data cant fill the full window size
    # length = len(ticker_segments)*window_size
    # if length != len(df):
    #     ticker_segments.append(df.iloc[length:])
    
    return ticker_segments

def generate_chart(df: pd.DataFrame, idx:int, ticker:str, save_dir:str, window_size: int = 90):
    os.makedirs(save_dir, exist_ok=True)

    mpf.plot(
        df, 
        type="candle", 
        style="yahoo", 
        figscale=1.5,
        savefig = os.path.join(save_dir, f"{ticker}_{idx}_to_{idx+window_size}.png")
    )
    
def main():
    TICKERS = [
        "AAPL",
        "GOOG",
        "COIN",
        "AMZN",
        "TQQQ",

    ]

    SAVE_DIR = "charts"
    get_and_process_data(TICKERS, SAVE_DIR, enable_charts=True)

# if __name__ == "__main__":
#     main()