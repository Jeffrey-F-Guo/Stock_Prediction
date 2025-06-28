from data_processing import get_and_process_data
from chart_data import ChartDataset
from baseline_lstm import StockLSTM
import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
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
    train_input_tensor = torch.from_numpy(train_input).float()
    train_target_tensor = torch.from_numpy(train_target).float()
    model = StockLSTM()
    optimizer = optim.AdamW(model.parameters())

    chart_data = ChartDataset(train_input_tensor, train_target_tensor)
    train_loader = DataLoader(chart_data, batch_size=batch_size, shuffle=True, num_workers=2)
    
    for epoch in range(epochs):

        avg_train_loss = train(train_loader, model, optimizer)
        print("="*10, f"train loss: {avg_train_loss}", "="*10, "\n", end="")
        if (epoch+1) % report_freq == 0:
            avg_dev_loss = evaluate(train_loader, model)
            print("="*10, f"dev loss: {avg_dev_loss}", "="*10, "\n", end="")
            # save_model()


def train(train_loader: DataLoader, model, optimizer):
    """training loop for one epoch"""
    model.train()
    total_loss = 0

    for inputs,targets in tqdm(train_loader, desc="training"):
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = F.mse_loss(outputs, targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    return total_loss/len(train_loader)

    

def evaluate(dev_loader: DataLoader, model):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, targets in tqdm(dev_loader, desc="Evaluating"):
            predictions = model(inputs)
            loss = F.mse_loss(predictions, targets)
            total_loss += loss.item()
    avg_loss = total_loss/len(dev_loader)
    return avg_loss

def save_model():
    ...