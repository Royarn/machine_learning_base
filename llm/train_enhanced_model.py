# check device
import torch
# load dataset
from MyDataset import MyDataset
# use bert tokenizer
from transformers import BertTokenizer
# use customized model
from Bert_customized import CustomizedModel
# load dataloader
from torch.utils.data import DataLoader
# load local config
from config import Cnf

def collate_fn(data):
    # sentences = [i[0] for i in data]
    sentences = [d[0] for d in data]
    # labels = [i[1] for i in data]
    labels = [i[1] for i in data]
    # encode sentences
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
    # convert labels to tensor
    labels = torch.LongTensor(labels)
    return input_ids, attention_mask, token_type_ids, labels

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using Device : ", DEVICE)

# load model and tokenizer
cache_dir = Cnf('bert')['local_dir']
tokenizer = BertTokenizer.from_pretrained(cache_dir)
model = CustomizedModel()

train_dataset = MyDataset("train")

train_loader = DataLoader(
    # load dataset
    train_dataset,
    # batch size
    batch_size=500,
    # shuffle data
    shuffle=True,
    # encode functions --usually, it depends on the tokenizer which you use
    collate_fn = collate_fn
)

# define loss function
loss_func = torch.nn.CrossEntropyLoss()
# define optimizer
optimizer = torch.optim.Adam(model.parameters())
model.to(DEVICE)
# begin training
EPOCH = 10
for epoch in range(EPOCH):
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
        # move data to device --includes GPU and CPU
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        labels = labels.to(DEVICE)
        # execute forward propagation
        output = model(input_ids, attention_mask, token_type_ids)
        # calculate loss
        loss = loss_func(output, labels)
        # clear gradients
        optimizer.zero_grad()
        # calculate gradients
        loss.backward()
        # update parameters
        optimizer.step()
        if i % 5 == 0:
            output = output.argmax(dim=1)
            accuracy = (output == labels).sum().item() / len(labels)
            # print epoch, loss, and calculate accuracy
            print("epoch:", epoch, "batch_no:", i, "loss:", loss.item(), "accuracy:", accuracy)
    # save model weights after every epoch
    torch.save(model.state_dict(), "params1/{}_bert.pth".format(epoch))
    print("save model weights after every epoch, and the location is :", "params/{}_bert.pth".format(epoch))