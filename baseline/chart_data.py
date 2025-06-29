# dataset class for dataloader
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
class ChartDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx:int):
        # semi-stateful LSTM approach -> Later. Move to different file
        # data will be features x tickers x total sequence length.
        # each time dataloader gets an item, it will be a features x mb x seq_window chunk of the data
        # travels across the row until those mb sequences of stock data are done.
        # then moves to the start of the next mb group of stock sequences in the data


        # stateless LSTM.
        # loaded data will be mb chunks, with each chunk being dimension seq_len x num_features
        input_seq, target = self.inputs[idx], self.targets[idx]

        # normalize 
        scaler = StandardScaler()
        norm_input = scaler.fit_transform(input_seq)

        return torch.from_numpy(norm_input).float(), target
    