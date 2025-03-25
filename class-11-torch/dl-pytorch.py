import torch
import torch.nn as nn

# 定义一个简单的两层神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleNet()
# 定义输入数据和真实标签
input_data = torch.randn(1, 10)
target = torch.randn(1, 1)

# 前向传播
output = model(input_data)

# 定义损失函数
criterion = nn.MSELoss()
# 计算损失
loss = criterion(output, target)

# 反向传播
loss.backward()

# 查看梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"Gradient of {name}: {param.grad}")