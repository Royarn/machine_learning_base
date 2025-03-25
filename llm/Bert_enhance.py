# enhance Bert model
import torch
from transformers import BertModel

# detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# local bert model
cache_dir = r"C:\Users\T290228H\app\langchain-base\llm\models\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
pre_trained = BertModel.from_pretrained(cache_dir).to(DEVICE)

class CustomizedModel(torch.nn.Module):
    def __init__(self):
        super(CustomizedModel, self).__init__()
        # original bert model
        print("Initialize original model : ", pre_trained)
        # define another linear layer
        self.incrementalFT = torch.nn.Linear(768, 2)
        # customize model
        print("Initialize custom model : ", self.incrementalFT)

    # forward execution of model
    def forward(self, input_ids, attention_mask, token_type_ids):
        # with no_grad()
        with torch.no_grad():
            output = pre_trained(input_ids, attention_mask, token_type_ids)
        # hidden state
        output = self.incrementalFT(output.last_hidden_state[:, 0])
        return output

# 创建模型实例
model = CustomizedModel().to('cpu')
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