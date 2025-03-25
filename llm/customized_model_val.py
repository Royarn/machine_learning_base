# okay, let's validate the model
# always, detect device
import torch
# load customized model and tokenizer
from Bert_enhance import CustomizedModel
from transformers import BertTokenizer
from torch.utils.data import DataLoader
# load customized dataset
from MyData import MyDataset


def collate_fn(data):
    sentences = [i[0] for i in data]
    label = [i[1] for i in data]
    encode_result = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=sentences,
        # 当句子长度大于max_length(上限是model_max_length)时，截断
        truncation=True,
        # check tokenizers config
        max_length=512,
        # 一律补0到max_length
        padding="max_length",
        # 可取值为tf,pt,np,默认为list
        return_tensors="pt",
        # 返回序列长度
    )
    input_ids = encode_result["input_ids"]
    attention_mask = encode_result["attention_mask"]
    token_type_ids = encode_result["token_type_ids"]
    labels = torch.LongTensor(label)
    return input_ids, attention_mask, token_type_ids, labels


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current machine supports : ", DEVICE)

val_dataset = MyDataset("Validation")

tokenizer = BertTokenizer.from_pretrained(r"C:\Users\T290228H\app\langchain-base\llm\models\bert-base-chinese")
model = CustomizedModel().to(DEVICE)

# load torch dataLoader
val_loader = DataLoader(
    # load dataset
    dataset=val_dataset,
    # batch size
    batch_size=50,
    # shuffle data
    shuffle=True,
    # encode functions --usually, it depends on the tokenizer which you use
    collate_fn=collate_fn
)

# begin to validate
correct = 0
correct_num = 0

model.load_state_dict(torch.load(r"C:\Users\T290228H\app\langchain-base\llm\params1\8_bert.pth", map_location=DEVICE))
# set model to eval mode
model.eval()
for i, (input_ids, attention_mask, token_type_ids, label) in enumerate(val_loader):
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    token_type_ids = token_type_ids.to(DEVICE)
    label = label.to(DEVICE)
    output = model(input_ids, attention_mask, token_type_ids)
    output = output.argmax(dim=1)
    # calculate accuracy
    correct = (output == label).sum().item()
    correct_num += correct
    # print current batch accuracy
    print("Current batch accuracy: ", correct / len(label))
print("Validation accuracy: ", correct / correct_num)
