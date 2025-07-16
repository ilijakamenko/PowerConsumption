from datasets import load_dataset
from torch.utils.data import Dataset


# Custom Dataset Class
class IMDBDataset(Dataset):
    def __init__(self):
        dataset = load_dataset("imdb")
        train_dataset = dataset["train"]
        self.texts: list[str] = train_dataset["text"]
        self.labels: list[str] = train_dataset["label"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
        return label, text
