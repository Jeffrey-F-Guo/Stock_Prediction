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

# final storage for data. will be 1 x (total number of 90-day sequences across all stocks)
# each entry will be a ((90x5), 1) a 90-day sequence of ochlv data and the target prediction for the 91st day--the next day
# tickers will be intermingled which is fine because we're treating them as general stock data instead of ticker-specific data
# dataloader will be able to shuffle and randomly pull 90-day chunks from this
SEGMENTED_DATA = [] 


def data_split(train_split:int, dev_split:int, inputs_df:pd.DataFrame, targets_df:pd.DataFrame):
    '''given two dataframes inputs and target containing all ticker OHLCV data. Shapes will be features x tickers x time length'''
    train_idx = int(len(inputs_df) * train_split)
    dev_idx = int(len(inputs_df) * dev_split)

    train_inputs, train_targets =  inputs_df.iloc[:train_idx], targets_df.iloc[:train_idx]
    dev_inputs, dev_targets = inputs_df.iloc[train_idx:dev_idx], targets_df.iloc[train_idx:dev_idx]
    test_inputs, test_targets = inputs_df.iloc[dev_idx:], targets_df.iloc[dev_idx:]

    return {
        "train": (train_inputs, train_targets),
        "dev": (dev_inputs, dev_targets),
        "test": (test_inputs, test_targets),
    }

def get_and_process_data(tickers:List, save_dir:str):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)


    ticker_dict = {}
    window_size = 90

    for tick in tickers:
        print(tick)
        ticker_obj, ticker_df = pull_data(tick)
        # calculate_macd(ticker_df)
        # calculate_bollinger(ticker_df)
        # calculate_bin_label(ticker_df)
        calculate_delta_p(ticker_df)

        ticker_dict[tick] = (ticker_obj, ticker_df)
        ticker_segments = segment_data(ticker_df)

        # commented out we dont have to regenerate the same images for every run
        # for k,v in ticker_dict.items():
        #     ticker_df = v[1]
        #     for i in range(0, len(ticker_df), window_size):
        #         segment_df = segment_data(ticker_df, i)
        #         generate_chart(segment_df, i, k, save_dir)
        SEGMENTED_DATA.extend(ticker_segments)


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
    if "Close_delta" not in df.columns:
        df["Close_delta"] = df["Close"].diff()
    if "Percent" not in df.columns:
        # df["Percent"] = df["Close_delta"]/df["Close"].shift(1) * 100
        df["Percent"] = df["Close"].pct_change() * 100
        # shift labels back by one day to simulate prediction. Each day's condition aim to predict the up/down status
        # of the next day
        df["Percent"] = df["Percent"].shift(1)
        df.ffill(inplace=True)

def extract_targets(df:pd.DataFrame) -> Tuple[pd.DataFrame, np.float32]:
    '''extract the targets of a single ticker. returned outputs will be concated to a main input and target dataframe'''
    columns = ["Open", "Close", "High", "Low", "Volume", "Percent"]
    if any(col not in df.columns for col in columns):
        print("input dataframe does not match expected format")
        return None, None

    # precondition that percent has been aggregated and shifted properly already
    inputs = df.drop("Percent")
    target = df["Percent"][:-1]

    return (inputs.to_numpy(), target)

def segment_data(df: pd.DataFrame, window_size: int=90):
    # 3 options for sampling items:
    # 1. separate non-overlapping chunks of size seq_len
    # 2. sliding window stride 1
    # 3. sliding window stride k where k != 1
    ticker_segments = np.array([extract_targets(df.iloc[idx:idx+window_size]) for idx in range(0, len(df), window_size)])

    # unecessary. python slicing handles this internally!
    # add the most recent data cant fill the full window size
    # length = len(ticker_segments)*window_size
    # if length != len(df):
    #     last_idx = len(df) - length
    #     ticker_segments.append(df.iloc[last_idx:])
    
    return ticker_segments

def generate_chart(df: pd.DataFrame, idx:int, ticker:str, save_dir:str, window_size: int = 30):
    os.makedirs(save_dir, exist_ok=True)

    mpf.plot(
        df, 
        type="candle", 
        style="yahoo", 
        figscale=1.5,
        savefig = os.path.join(save_dir, f"{ticker}_{idx}_to_{idx+window_size}.png")
    )
    


