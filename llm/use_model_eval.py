# load local tokenizer
from transformers import BertTokenizer
from bert_model import Model
from MyData import MyDataset
from torch.utils.data import DataLoader
import torch

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#定义训练的轮次(将整个数据集训练完一次为一轮)
EPOCH = 10

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
        return_length=True
    )
    # 字符编码后的ids
    input_ids = encode_result["input_ids"]
    # 字符编码后的mask
    attention_mask = encode_result["attention_mask"]
    # token type id
    token_type_ids = encode_result["token_type_ids"]
    # label
    labels = torch.LongTensor(label)
    return input_ids, attention_mask, token_type_ids, labels


names = ['Positive', 'Negative']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model_name = r"C:\Users\T290228H\app\langchain-base\llm\models\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"

# load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = Model()

# load customized data
train_dataset = MyDataset("train")

# use torch dataLoader
train_loader = DataLoader(
    # customized data
    dataset=train_dataset,
    # training batch size
    batch_size=10,
    # shuffle data
    shuffle=True,
    # drop last batch if not enough data
    drop_last=True,
    # collate function  --编码函数 应该是bert 的编码函数
    collate_fn=collate_fn
)


if __name__ == '__main__':
    # device type
    print(DEVICE)

    # load model to device
    model = model.to(DEVICE)
    # load optimizer
    optimizer = torch.optim.AdamW(model.parameters())
    # load loss function
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            # execute forward propagation
            out = model(input_ids, attention_mask, token_type_ids)
            loss = loss_func(out, labels)
            # execute backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # every 5 batch print loss
            if i % 5 == 0:
                out = out.argmax(dim=1)
                # calculate accuracy
                acc = (out == labels).sum().item() / len(labels)
                print("epoch:{},batch:{},loss:{},acc:{}".format(epoch, i, loss.item(), acc))

        # every epoch save model
        torch.save(model.state_dict(), "params/{}_bert.pth".format(epoch))