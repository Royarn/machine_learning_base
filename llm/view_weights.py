import torch
import torch.nn as nn

# 定义一个简单的示例模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# 创建模型实例
model = SimpleModel().to('cpu')
model_path = r'C:\Users\T290228H\app\langchain-base\llm\params1\8_bert.pth'

try:
    state_dict = torch.load(model_path, map_location='cpu')
    # 将加载的权重应用到模型上
    model.load_state_dict(state_dict)
    print("成功将权重加载到模型中。")
except FileNotFoundError:
    print(f"错误: 未找到 {model_path} 文件。")
except Exception as e:
    print(f"错误: 加载模型权重时出现未知错误: {e}")