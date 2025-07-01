import torch
from baseline_lstm import StockLSTM
import torch.optim as optim
import os
from data_processing import stock_map

def load_model(filename='models/checkpoint.pth', learning_rate=0.001):
    model = StockLSTM()
    optimizer = optim.AdamW(model.parameters(), learning_rate)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file {filename} not found.")
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded model and optimizer state from {filename}")
    return model, optimizer

def eval_model():
    model, optimizer = load_model()

    # load dev data
    for tick in tickers:

