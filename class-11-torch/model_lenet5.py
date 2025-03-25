import torch.nn as nn
import torch

# 定义leNet-5 模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 第一层卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # 第一层池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # 第二层卷积层
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 第二层池化层 --池化的目的是为了减少参数数量，提高模型性能
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 全连接层
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)


    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x