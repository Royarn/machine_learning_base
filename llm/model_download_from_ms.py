#模型下载
from modelscope import snapshot_download

# local dir
local_dir = r"C:\Users\T290228H\app\langchain-base\llm\models"

model = snapshot_download('tiansz/bert-base-chinese', cache_dir=local_dir)

print(model)

print('Model has been loaded to local')