"""
encoding = 'utf-8'
author: Vico Zhang
此文件生成神经网络，为 LeNet-5 的改动版本。
More information: https://github.com/VicoZhang/Project_0704.git
"""

import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3)),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.ReLU()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU()
        )
        self.Connection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=4*16, out_features=16),
            nn.Linear(in_features=16, out_features=4)
        )

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Connection(x)
        return x


if __name__ == '__main__':
    input_test = torch.reshape(torch.ones(23*23), (-1, 1, 23, 23))
    net = Net()
    out_test = net(input_test)
    print(out_test.shape)
    print("测试通过")

