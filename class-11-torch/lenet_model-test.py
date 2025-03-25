import torch
# import torch.nn.functional as F -- full connection
import torch.nn as nn
# optimizer
import torch.optim as optimizer
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 检查gpu -cuda 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

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


# 加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

# 定义数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# 定义模型，损失函数，优化器
# 模型加载到CPU
model = LeNet5()
# 损失函数 --交叉熵 / softmax
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optimizer.SGD(model.parameters(), lr=0.01, momentum=0.9)
# 训练轮次
epochs = 20
for epoch in range(epochs):
    # 初始化变量
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # 梯度清零
        # 在每次参数更新之前，需要将模型参数的梯度清零。
        # 因为 PyTorch 中梯度是累积的，如果不清零，梯度会不断累加，导致参数更新出现错误。
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 损失函数
        loss = criterion(outputs, labels)

        # 反向传播计算梯度
        # 调用 loss.backward() 方法进行反向传播，计算损失函数关于模型参数的梯度。
        # 这一步会根据计算图，利用链式法则自动计算每个参数的梯度，并将梯度存储在参数的 .grad 属性中。
        loss.backward()

        # 更新模型参数
        # 调用 optimizer.step() 方法，根据之前计算得到的梯度，使用优化器（如 SGD、Adam 等）来更新模型的参数。
        # 优化器会根据预设的学习率和梯度信息，对模型的参数进行调整，使得损失函数逐渐减小。
        optimizer.step()
        # 打印损失值
        running_loss += loss.item()
        if i % 100 == 99:
            # 打印损失值/ 损失/精度
            print('[%d, %5d] loss： %.3f' %(epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

# 训练完成
print('Finished Training')
# 保存模型
torch.save(model.state_dict(), 'lenet5.pth')


# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))