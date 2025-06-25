from data_processing import get_and_process_data
from chart_data import ChartDataset
from baseline_lstm import StockLSTM
import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
def main():

    TICKERS = [
        "AAPL",
        "GOOG",
        "COIN",
        "AMZN",
        "TQQQ",

    ]

    SAVE_DIR = "charts"

    # hyperparameters
    epochs = 50
    batch_size = 32
    report_freq = 10
    

    # for now assume this returns a numpy array
    get_and_process_data(TICKERS, SAVE_DIR)

    model = StockLSTM()
    optimizer = optim.AdamW(model.parameters())
    for epoch in range(epochs):
        chart_data = ChartDataset(train_input, train_target)
        train_loader = DataLoader(chart_data, batch_size=batch_size, shuffle=True, num_workers=2)
        train(batch_size, train_loader, model)
        if (epoch+1) % 10 == 0:
            evaluate()
            save_model()


def train(mb:int, train_loader:torch.Tensor, model, optimizer):
    """training loop for one epoch"""

    for inputs,targets in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = nn.MSELoss(outputs, targets)

        
        
        loss.backward()
        optimizer.step()

    

def evaluate():
    ...

def save_model():
