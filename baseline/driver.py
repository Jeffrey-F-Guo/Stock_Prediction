from data_processing import get_and_process_data
from chart_data import ChartDataset
from baseline_lstm import StockLSTM
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os

def main():
    print("STARTED")
    # TODO: change this to read from file to avoid length string
    # TICKERS = [
    #     "AAPL",
    #     "GOOG",
    #     "COIN",
    #     "AMZN",
    #     "TQQQ",
    #     "META",
    #     "MSFT",
    #     "NVDA",
    #     "PLTR",
    #     "LCID",
    #     "NKE",
    #     "SOFI",
    #     "TSLA",
    #     "INTC",
    #     "RKLB",
    #     "SNAP",
    #     "HIMS",
    #     "AMD",
    #     "ENPH",
    #     "RBLX",
    #     "PYPL",
    #     "TMC",
    #     "TSM"

    # ]
    TICKERS = ['TCEHY', 'NFLX', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'BIDU', 'GOOG', 'AAPL', 'INTC']

    SAVE_DIR = "charts"

    # hyperparameters
    epochs = 50
    batch_size = 32
    report_freq = 10
    learn_rate = 0.001
    
    model = StockLSTM()
    optimizer = optim.AdamW(model.parameters(), learn_rate)

    data_dict = get_and_process_data(TICKERS, SAVE_DIR, enable_charts=False)

    train_inputs = data_dict["train"]["inputs"]
    train_targets = data_dict["train"]["targets"]
    dev_inputs = data_dict["dev"]["inputs"]
    dev_targets = data_dict["dev"]["targets"]

    # train_input_tensor = torch.from_numpy(train_inputs).float()
    # train_target_tensor = torch.from_numpy(train_targets).float()

    # dev_input_tensor = torch.from_numpy(dev_inputs).float()
    # dev_target_tensor = torch.from_numpy(dev_targets).float()

    train_chart_data = ChartDataset(train_inputs, train_targets)
    train_loader = DataLoader(train_chart_data, batch_size=batch_size, shuffle=True)

    dev_chart_data = ChartDataset(dev_inputs, dev_targets)
    dev_loader = DataLoader(dev_chart_data, batch_size=batch_size, shuffle=True)

  
    
    for epoch in range(epochs):
        avg_train_loss = train(train_loader, model, optimizer)
        print("="*10, f"train loss: {avg_train_loss}", "="*10, "\n", end="")
        if (epoch+1) % report_freq == 0:
            avg_dev_loss = evaluate(dev_loader, model)
            print("="*10, f"dev loss: {avg_dev_loss}", "="*10, "\n", end="")
            save_model(model, optimizer, "checkpoint.pth")


def train(train_loader: DataLoader, model, optimizer):
    """training loop for one epoch"""
    model.train()
    total_loss = 0

    for inputs,targets in train_loader:
        optimizer.zero_grad()
        outputs = torch.squeeze(model(inputs))
        print(f"output: {outputs}, target: {targets}")
        loss = F.mse_loss(outputs, targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    return total_loss/len(train_loader)
    

def evaluate(dev_loader: DataLoader, model):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, targets in (dev_loader):
            predictions = torch.squeeze(model(inputs))
            loss = F.mse_loss(predictions, targets)
            total_loss += loss.item()
    avg_loss = total_loss/len(dev_loader)
    return avg_loss

def save_model(model, optimizer, filename:str):
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)

if __name__ == "__main__":
    main()