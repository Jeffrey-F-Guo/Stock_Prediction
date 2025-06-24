from data_processing import get_and_process_data
import chart_data 
import baseline_lstm

def main():

    TICKERS = [
        "AAPL",
        "GOOG",
        "COIN",
        "AMZN",
        "TQQQ",

    ]

    SAVE_DIR = "charts"

    get_and_process_data(TICKERS, SAVE_DIR)

