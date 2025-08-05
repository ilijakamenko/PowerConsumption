# ---------------------------------------------------------------------------------------------
from datasets import load_dataset
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from device_info import DeviceInfo
from results import TestResult
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

try:
    import torch_npu
except:
    pass
# ---------------------------------------------------------------------------------------------


def text_pipeline(tokenizer, text: str):
    return tokenizer.encode(text, truncation=True, max_length=512)


def get_collate_fn(tokenizer):
    def collate_fn(batch):
        label_list, text_list = [], []
        for item in batch:
            _label = item["label"]
            _text = item["text"]
            label_list.append(torch.tensor(_label))
            processed_text = torch.tensor(
                text_pipeline(tokenizer, _text), dtype=torch.int64
            )
            text_list.append(processed_text)
        return (
            torch.tensor(label_list),
            pad_sequence(text_list, batch_first=True),
        )

    return collate_fn


def evaluate_model(model, test_loader, device: str):
    total_correct = 0
    total_loss = 0
    total_samples = 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for _, (labels, texts) in enumerate(test_loader):
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = loss_fn(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            correct = predicted.eq(labels).sum().item()

            total_correct += correct
            total_loss += loss.item() * texts.size(0)
            total_samples += texts.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return avg_loss, accuracy


def test(
    model: nn.Module, device_info: DeviceInfo, num_workers: int, batch_size: int
) -> TestResult:
    model.eval()

    dataset = load_dataset("imdb")
    test_dataset = dataset["test"]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    collate_fn = get_collate_fn(tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )  # for a100 add num_workers>1
    avg_loss, accuracy = evaluate_model(model, test_loader, device_info.device)
    return TestResult(accuracy=accuracy, avg_loss=avg_loss)
