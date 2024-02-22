import torch
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=False)  # 28x28x1 -> 26x26x32
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, bias=False
        )  # 26x26x32 -> 24x24x64
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, bias=False
        )  # 24x24x64 -> 22x22x128
        self.conv4 = nn.Conv2d(
            128, 256, kernel_size=3, bias=False
        )  # 22x22x128 -> 20x20x
        self.fc1 = nn.Linear(4096, 50, bias=False)
        self.fc2 = nn.Linear(50, 10, bias=False)

    def forward(self, x):
        x = F.relu(
            self.conv1(x), 2
        )  # create a relu with thresh 2, any values greater than 2 is retained # 28x28x1 -> 26x26x32
        x = F.relu(
            F.max_pool2d(self.conv2(x), 2)
        )  # 26x26x32 -> 24x24x64  ->(MP) 12x12x64
        x = F.relu(self.conv3(x), 2)  # 12x12x64 -> 10x10x128
        x = F.relu(
            F.max_pool2d(self.conv4(x), 2)
        )  # 10x10x128 -> 8x8x256 ->(MP) 4x4x256
        x = x.view(-1, 4096)  # Flatten to 4096
        x = F.relu(self.fc1(x))  # 4096 -> 50
        x = self.fc2(x)  # 50 -> 10
        return F.log_softmax(x, dim=1)  # Softmax to output
