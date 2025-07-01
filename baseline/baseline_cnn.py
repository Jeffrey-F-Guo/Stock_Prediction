import torch
import torch.nn as nn


class StockCNN(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout_rate = dropout
        self.h1_dim = 32
        self.h2_dim = 64
        self.h3_dim = 128
        self.kernel_size = 5

        self.cnn = nn.Sequential(
            nn.Conv2d(1, self.h1_dim, kernel_size=self.kernel_size, padding=1),
            nn.BatchNorm2d(self.h1_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),

            nn.Conv2d(self.h1_dim, self.h2_dim, kernel_size=self.kernel_size, padding=1),
            nn.BatchNorm2d(self.h2_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),

            nn.Conv2d(self.h2_dim, self.h3_dim, kernel_size=self.kernel_size, padding=1),
            nn.BatchNorm2d(self.h3_dim),
            nn.BatchNorm2d(self.h2_dim),
            nn.Dropout(self.dropout_rate),
        )

        self.fc_input_dim =  self.h2_dim
        self.h1_dim = 64
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
        x = self.cnn(x)
        x = self.dnn(x)
        return x



