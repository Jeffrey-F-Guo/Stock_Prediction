# dataset class for dataloader
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch

class ChartDataset(Dataset):
    def __init__(self, inputs:torch.Tensor, targets:torch.Tensor, seq_len:int):
        self.inputs = inputs
        self.targets = targets
        self.seq_len = seq_len

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx:int):
        # semi-stateful LSTM approach -> Later. Move to different file
        # data will be features x tickers x total sequence length.
        # each time dataloader gets an item, it will be a features x mb x seq_window chunk of the data
        # travels across the row until those mb sequences of stock data are done.
        # then moves to the start of the next mb group of stock sequences in the data


        # stateless LSTM.
        # loaded data will be mb chunks, with each chunk being seq_len = 90 days long
        input_seq, target_seq = self.inputs[idx], self.targets[idx]

        # normalize 
        scaler = StandardScaler()
        norm_input = scaler.fit_transform(input_seq)
        norm_target = scaler.fit_transform(target_seq)

        return norm_input, norm_target
    