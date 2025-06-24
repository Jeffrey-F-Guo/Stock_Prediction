# dataset class for dataloader
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

class ChartDataset(Dataset):
    def __init__(self, inputs:torch.Tensor, targets:torch.Tensor, seq_len:int):
        self.inputs = inputs
        self.targets = targets
        self.seq_len = seq_len

    def __len__(self):
        return len(self.inputs)//self.seq_len
    
    def __getitem__(self, idx:int):
        # data will be features x tickers x total sequence length.
        # each time dataloader gets an item, it will be a features x mb x seq_window chunk of the data
        # travels across the row until those mb sequences of stock data are done.
        # then moves to the start of the next mb group of stock sequences in the data


        # 3 options for sampling items:
        # 1. separate non-overlapping chunks of size seq_len
        # 2. sliding window stride 1
        # 3. sliding window stride k where k != 1

        # sample strategy 1
        start = idx*self.seq_len
        end = start+self.seq_len

        if end > len(self.inputs):
            raise IndexError("End index out of bounds")
        input_seq, target_seq = self.inputs[start:end], self.targets[start:end]

        # normalize 
        scaler = StandardScaler()
        norm_input = scaler.fit_transform(input_seq)
        norm_target = scaler.fit_transform(target_seq)

        return norm_input, norm_target
    