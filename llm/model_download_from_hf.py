from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# local dir
local_dir = r"C:\Users\T290228H\app\langchain-base\llm\models"
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', cache_dir=local_dir)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir=local_dir)

print(model)
print('Model has been loaded to local')