import torch
from model_lenet5 import LeNet5

# 加载模型
model = LeNet5()
# 加载模型参数
model.load_state_dict(torch.load('./lenet5.pth'))

# 打印模型结构
print(model)