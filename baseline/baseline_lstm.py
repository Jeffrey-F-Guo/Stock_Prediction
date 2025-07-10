import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    def __init__(self, input_dim=5, dropout=0.1):
        super().__init__()
        self.dropout_rate = dropout
        self.input_dim = input_dim
        self.h1_dim = 32
        self.h2_dim = 128
        self.h3_dim = 256

        self.lstm1 = nn.LSTM(self.input_dim, self.h1_dim, batch_first=True)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        # self.norm1 = nn.LayerNorm(self.h1_dim)

        self.lstm2 = nn.LSTM(self.h1_dim, self.h2_dim, batch_first=True)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        # self.norm2 = nn.LayerNorm(self.h2_dim)

        self.lstm3 = nn.LSTM(self.h2_dim, self.h3_dim, batch_first=True)
        self.dropout3 = nn.Dropout(self.dropout_rate)
        # self.norm3 = nn.LayerNorm(self.h3_dim)

        self.fc_input_dim =  self.h3_dim
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
        x = self.dropout1(x)
        # x = self.norm1(x)

        x, (h_n2, c_n2) = self.lstm2(x)
        x = self.dropout2(x)
        # x = self.norm2(x)

        x, (h_n3, c_n3) = self.lstm3(x)
        x = self.dropout3(x)
        # x = self.norm3(x)

        x = x[:, -1: :]
        x = self.dnn(x)
        return x



