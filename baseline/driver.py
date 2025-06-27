from data_processing import get_and_process_data
from chart_data import ChartDataset
from baseline_lstm import StockLSTM
import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
def main():

    # TODO: change this to read from file to avoid length string
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
    

    
    train_input, train_target = get_and_process_data(TICKERS, SAVE_DIR)
    train_input_tensor = torch.from_numpy(train_input)
    train_target_tensor = torch.from_numpy(train_target)
    model = StockLSTM()
    optimizer = optim.AdamW(model.parameters())
    for epoch in range(epochs):
        chart_data = ChartDataset(train_input_tensor, train_target_tensor)
        train_loader = DataLoader(chart_data, batch_size=batch_size, shuffle=True, num_workers=2)
        train(batch_size, train_loader, model)
        if (epoch+1) % 10 == 0:
            evaluate()
            # save_model()


def train(train_loader:torch.Tensor, model, optimizer):
    """training loop for one epoch"""
    model.train()
    for inputs,targets in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = nn.MSELoss(outputs, targets)

        loss.backward()
        optimizer.step()

    

def evaluate(dev_loader:torch.Tensor, model, optimizer):
    model.eval()
    with 

def save_model():
    ...