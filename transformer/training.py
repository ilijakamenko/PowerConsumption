import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import time

from device_info import DeviceInfo
from imdb import IMDBDataset
from power import measure_power
from results import PowerMeasurement, TrainingResults
from transformer.model import TransformerModel

try:
    import torch_npu
except:
    pass

# ---------------------------------------------------------------------------------------------


# Function to preprocess data
def text_pipeline(tokenizer, text):
    return tokenizer.encode(text, truncation=True, max_length=512)


# Collate function to pad sequences
def get_collate_fn(tokenizer):
    def collate_fn(batch):
        label_list, text_list = [], []
        for _label, _text in batch:
            label_list.append(torch.tensor(_label))
            processed_text = torch.tensor(
                text_pipeline(tokenizer, _text), dtype=torch.int64
            )
            text_list.append(processed_text)
        return torch.tensor(label_list), pad_sequence(text_list, batch_first=True)

    return collate_fn


def train(device_info: DeviceInfo) -> TrainingResults:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    collate_fn = get_collate_fn(tokenizer)

    train_data = IMDBDataset()
    train_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
    )  # for a100 add num_workers > 1

    # Initialize model, loss, and optimizer
    vocab_size = tokenizer.vocab_size
    embed_size = 128
    num_heads = 8
    hidden_dim = 256
    num_layers = 2
    output_size = 2  # Binary classification (positive/negative)

    model = TransformerModel(
        vocab_size, embed_size, num_heads, hidden_dim, num_layers, output_size
    ).to(device_info.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Specify baseline power consumption (in Watts)
    baseline_power = round(measure_power(device_info), 2)

    # Start training
    num_epochs = 5
    measurements = []

    for epoch in tqdm(range(num_epochs), desc="Training"):
        epoch_start_time = time.time()
        model.train()

        for _, (label, text) in enumerate(train_loader):
            text, label = text.to(device_info.device), label.to(device_info.device)
            optimizer.zero_grad()
            output = model(text)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            measurements.append(
                PowerMeasurement(
                    epoch=epoch,
                    elapsed_time=round(time.time() - epoch_start_time, 2),
                    power_consumption=round(
                        measure_power(device_info) - baseline_power, 2
                    ),
                )
            )

    return TrainingResults(model=model, measurements=measurements)
