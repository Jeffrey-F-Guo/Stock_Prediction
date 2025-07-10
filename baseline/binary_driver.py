# Binary LSTM driver for binary classification
from data_processing import get_and_process_data
from chart_data import ChartDataset
from baseline_lstm import StockLSTM
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np

def main():
    print("STARTED (Binary Classification)")
    TICKERS = [
        "AAPL",
        "GOOG",
        "COIN",
        "AMZN",
        "TQQQ",
        "META",
        "MSFT",
        "NVDA",
        "PLTR",
        "LCID",
        "NKE",
        "SOFI",
        "TSLA",
        "INTC",
        "RKLB",
        "SNAP",
        "HIMS",
        "AMD",
        "ENPH",
        "RBLX",
        "PYPL",
        "TMC",
        "TSM"

    ]
    SAVE_DIR = "charts"
    epochs = 50
    batch_size = 32
    report_freq = 10
    learn_rate = 0.001
    model = StockLSTM()
    optimizer = optim.AdamW(model.parameters(), learn_rate)

    data_dict = get_and_process_data(TICKERS, SAVE_DIR, enable_charts=False)

    def binarize_targets(targets):
        # 0 for no growth (<=0), 1 for growth (>0)
        return (targets > 0).astype(np.float32)

    train_inputs = data_dict["train"]["inputs"]
    train_targets = binarize_targets(data_dict["train"]["targets"])
    dev_inputs = data_dict["dev"]["inputs"]
    dev_targets = binarize_targets(data_dict["dev"]["targets"])

    train_chart_data = ChartDataset(train_inputs, train_targets)
    train_loader = DataLoader(train_chart_data, batch_size=batch_size, shuffle=True)

    dev_chart_data = ChartDataset(dev_inputs, dev_targets)
    dev_loader = DataLoader(dev_chart_data, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        avg_train_loss, train_acc = train(train_loader, model, optimizer)
        print("="*10, f"train loss: {avg_train_loss:.4f} acc: {train_acc:.4f}", "="*10, "\n", end="")
        if (epoch+1) % report_freq == 0:
            avg_dev_loss, dev_acc = evaluate(dev_loader, model)
            print("="*10, f"dev loss: {avg_dev_loss:.4f} acc: {dev_acc:.4f}", "="*10, "\n", end="")
            save_model(model, optimizer, f"checkpoint_binary_epoch{epoch+1}.pth")

def train(train_loader: DataLoader, model, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.BCEWithLogitsLoss()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = torch.squeeze(model(inputs))
        targets = targets.float()
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.numel()
    acc = correct / total if total > 0 else 0
    return total_loss/len(train_loader), acc

def evaluate(dev_loader: DataLoader, model):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for inputs, targets in dev_loader:
            outputs = torch.squeeze(model(inputs))
            targets = targets.float()
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.7).float()
            correct += (preds == targets).sum().item()
            total += targets.numel()
    acc = correct / total if total > 0 else 0
    avg_loss = total_loss/len(dev_loader)
    return avg_loss, acc

def save_model(model, optimizer, filename:str):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

if __name__ == "__main__":
    main() 