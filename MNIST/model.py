import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), 1, padding=1)
        self.maxpool = nn.MaxPool2d((2, 2), 2)
        self.linear1 = nn.Linear(7 * 7 * 64, 128)
        self.linear2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        pred = self.linear2(x)

        return pred



