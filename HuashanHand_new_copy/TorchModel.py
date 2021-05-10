import numpy as np
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F


class Conv2dNet(nn.Module):

    def __init__(self):
        super(Conv2dNet, self).__init__()
        self.conv1 = nn.Conv2d(5, 10, 7)
        self.pool = nn.MaxPool2d(5, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(20*10*10, 500)
        self.fc2 = nn.Linear(500, 125)
        self.fc3 = nn.Linear(125, 20)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.25)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
