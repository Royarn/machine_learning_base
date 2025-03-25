from transformers import BertTokenizer

local_dir = r"C:\Users\T290228H\app\langchain-base\llm\models\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"

tokenizer = BertTokenizer.from_pretrained(local_dir)

sents = ["白日依山尽，",
         "价格在这个地段属于适中, 附近有早餐店,小饭店, 比较方便,无早也无所"]

# encode the inputs
out = tokenizer.batch_encode_plus(
    # 输入文本内容
    batch_text_or_text_pairs=[sents[0], sents[1]],
    # 是否添加特殊标记
    add_special_tokens=True,
    # encode 最大长度，超出长度部分会被截断
    max_length=512,
    # 是否补齐
    padding='max_length',
    # 是否对输入进行截断
    truncation=True,
    # 返回张量类型
    return_tensors=None,
    # 是否返回attention mask
    return_attention_mask = True,
    # 是否返回token type ids
    return_token_type_ids = True,
    # 是否返回特殊标记的mask
    return_special_tokens_mask = True,
    # 返回序列长度
    return_length = True
)

#input_ids 就是编码后的词
#token_type_ids第一个句子和特殊符号的位置是0，第二个句子的位置1（）只针对于上下文编码
#special_tokens_mask 特殊符号的位置是1，其他位置是0
#length 编码之后的序列长度

for i, v in out.items():
    print(i, ":",  v)

#解码文本数据
print(tokenizer.decode(out["input_ids"][0]),tokenizer.decode(out["input_ids"][1]))