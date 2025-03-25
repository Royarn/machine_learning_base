#now lets try to evaluate this customized model
from Bert_enhance import CustomizedModel
from transformers import BertTokenizer
import torch
from MyData import MyDataset
from torch.utils.data import DataLoader

def collate_fn(data):
    sentences = [i[0] for i in data]
    label = [i[1] for i in data]
    # encode data
    encode_result = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=sentences,
        # when the sentence is longer than max_length(default is model_max_length), truncate it
        truncation=True,
        # check tokenizers config
        max_length=512,
        # padding to max_length with 0
        padding="max_length",
        # possible values: "pt", "np", "tf", "torch"
        return_tensors="pt",
        # return sequence length
        return_length=True
    )
    input_ids = encode_result["input_ids"]
    attention_mask = encode_result["attention_mask"]
    token_type_ids = encode_result["token_type_ids"]
    labels = torch.LongTensor(label)
    return input_ids, attention_mask, token_type_ids, labels

# firstly, detect the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using Device : ", DEVICE)

# load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(r"C:\Users\T290228H\app\langchain-base\llm\models\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")
model = CustomizedModel().to(DEVICE)

# load customized data
test_dataset = MyDataset("test")
test_loader = DataLoader(
    # load dataset
    dataset=test_dataset,
    # set batch size
    batch_size=50,
    # shuffle data
    shuffle=True,
    # set collate function  --it depends on the tokenizer which you use
    collate_fn=collate_fn,
    # drop last batch if not enough data
    drop_last=True
)

# load customized weight
model.load_state_dict(torch.load(r"C:\Users\T290228H\app\langchain-base\llm\params1\9_bert.pth", map_location=DEVICE))

# between the training and testing, we need to set the model to eval mode
model.eval()

# test the model
correct = 0
correct_num = 0
for i,(input_ids,attention_mask,token_type_ids,labels) in enumerate(test_loader):
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    token_type_ids = token_type_ids.to(DEVICE)
    labels = labels.to(DEVICE)
    # execute forward propagation
    out = model(input_ids, attention_mask, token_type_ids)
    # calculate average loss, average accuracy
    out = out.argmax(dim=1)
    correct += (out == labels).sum().item()
    correct_num += len(labels)
    # calculate current accuracy, loss
    acc = (out == labels).sum().item() / len(labels)
    print("Current Test Accuracy: ", acc)
print("Test Accuracy: ", correct / correct_num)