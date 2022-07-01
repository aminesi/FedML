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


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x