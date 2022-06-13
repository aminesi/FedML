import torch
from torch import nn
import torch.nn.functional as F


class CifarCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv2d_1 = nn.Conv2d(3, 32, 3, padding='same')
        self.conv2d_2 = nn.Conv2d(32, 32, 3)
        self.max_pool = nn.MaxPool2d(2)
        self.dropout_1 = nn.Dropout(.25)

        self.conv2d_3 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv2d_4 = nn.Conv2d(64, 64, 3)
        self.dropout_2 = nn.Dropout(.25)

        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(2304, 512)
        self.dropout_3 = nn.Dropout(.5)
        self.linear_2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv2d_1(x))
        x = F.relu(self.conv2d_2(x))
        x = self.max_pool(x)
        x = self.dropout_1(x)

        x = F.relu(self.conv2d_3(x))
        x = F.relu(self.conv2d_4(x))
        x = self.max_pool(x)
        x = self.dropout_2(x)

        x = self.flatten(x)
        x = F.relu(self.linear_1(x))
        x = self.dropout_3(x)
        x = F.softmax(self.linear_2(x), dim=1)

        return x
