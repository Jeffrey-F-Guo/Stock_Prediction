import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    def __init__(self, input_dim=5, dropout=0.1):
        super().__init__()
        self.dropout_rate = dropout
        self.input_dim = input_dim
        self.h1_dim = 64
        self.h2_dim = 512

        self.lstm1 = nn.LSTM(self.input_dim, self.h1_dim, batch_first=True, dropout=self.dropout_rate)
        self.norm1 = nn.LayerNorm(self.h1_dim)

        self.lstm2 = nn.LSTM(self.h1_dim, self.h2_dim, batch_first=True, dropout=self.dropout_rate) # Add dropout here too if desired
        self.norm2 = nn.LayerNorm(self.h2_dim)


        self.fc_input_dim =  self.h2_dim
        self.h1_dim = 128
        self.h2_dim = 32
        self.h3_dim = 1

        self.dnn = nn.Sequential(
            nn.Linear(self.fc_input_dim,self.h1_dim),
            nn.ReLU(),
            nn.Linear(self.h1_dim, self.h2_dim),
            nn.ReLU(),
            nn.Linear(self.h2_dim, self.h3_dim),
        )

    def forward(self, x):
        x, (h_n1, c_n1) = self.lstm1(x)
        x = self.norm1(x)

        x, (h_n2, c_n2) = self.lstm2(x)
        x = self.norm2(x)

        x = x[:, -1: :
              ]
        x = self.dnn(x)
        return x



