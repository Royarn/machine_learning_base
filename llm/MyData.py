from torch.utils.data import Dataset
from datasets import load_from_disk


class MyDataset(Dataset):
    def __init__(self, split):
        # load local dataset
        self.dataset = load_from_disk(r"C:\Users\T290228H\app\langchain-base\llm\data\ChnSentiCorp")
        if split == 'train':
            self.dataset = self.dataset["train"]
        elif split == 'test':
            self.dataset = self.dataset["test"]
        elif split == 'validation':
            self.dataset = self.dataset["validation"]
        else:
            print("Invalid type！")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text = self.dataset[item]['text']
        label = self.dataset[item]['label']
        return text, label

if __name__ == '__main__':
    dataset = MyDataset("test")
    for data in dataset:
        print(data)