import torch
import torch.nn as nn


class BinaryStockLSTM(nn.Module):
    def __init__(self, input_dim=5, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.input_dim = input_dim
        self.h1_dim = 64
        self.h2_dim = 512

        self.lstm = nn.Sequential(
            nn.LSTM(self.input_dim, self.h1_dim, dropout=dropout),
            nn.LayerNorm(self.h1_dim),
            nn.LSTM(self.h1_dim, self.h2_dim),
            nn.LayerNorm(self.h2_dim),
        )

        self.fc_input_dim = 512
        self.h1_dim = 128
        self.h2_dim = 32
        self.h3_dim = 1

        self.dnn = nn.Sequential(
            nn.Linear(self.fc_input_dim,),
            nn.ReLU(),
            nn.Linear(self.h1_dim, self.h2_dim),
            nn.ReLU(),
            nn.Linear(self.h2_dim, self.h3_dim),
        )

    def forward(self, x):
        x = self.lstm(x)
        x = self.dnn(x)
        return x



def train():
    ...
def main():



