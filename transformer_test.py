#---------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import math
try:
    import torch_npu
except:
    pass
#---------------------------------------------------------------------------------------------

device_type = "cuda"  # specify platform ['cuda', 'npu', 'cpu']
device_number = "0"
device = device_type + ":" + device_number

# Load the model
model_file = "models/" + os.path.basename(__file__).split('_')[0] + '_' + device_type + '.pt'
print(f"Loading model from {model_file}")

# Define the Transformer model (same as the one used during training)
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, output_size, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        transformer_layer = nn.TransformerEncoderLayer(embed_size, num_heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers)
        self.fc = nn.Linear(embed_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.embedding(src) * math.sqrt(src.size(1))  # Scaling
        embedded = self.pos_encoder(embedded)
        transformer_out = self.transformer_encoder(embedded)
        output = self.fc(self.dropout(transformer_out.mean(dim=1)))  # Aggregate over sequence
        return output

# Load the trained Transformer model
model = torch.load(model_file, map_location=device)
model.eval()  # Set model to evaluation mode

# Tokenizer and vocabulary (same as used during training)
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# Reload the IMDB dataset for building vocabulary
train_iter, test_iter = IMDB(split=('train', 'test'))
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Preprocessing pipelines
def text_pipeline(text):
    return vocab(tokenizer(text))

def label_pipeline(label):
    return 1 if label == "pos" else 0

# Collate function for padding sequences
def collate_batch(batch):
    label_list, text_list, raw_texts = [], [], []
    for (_label, _text) in batch:
        label_list.append(torch.tensor(label_pipeline(_label)))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        raw_texts.append(_text)  # Store raw text for printing
    return torch.tensor(label_list), pad_sequence(text_list, batch_first=True), raw_texts

# Prepare the test dataset
test_loader = DataLoader(list(test_iter), batch_size=64, shuffle=True, collate_fn=collate_batch)  #for a100 add num_workers>1

# Define the criterion (loss function)
criterion = nn.CrossEntropyLoss()

# Evaluate the model on the test set and print raw text samples
def evaluate_model(model, test_loader):
    total_correct = 0
    total_loss = 0
    total_samples = 0
    num_samples_to_print = 5
    printed_samples = 0

    with torch.no_grad():  # No gradient computation during evaluation
        for batch_idx, (labels, texts, raw_texts) in enumerate(test_loader):
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct = predicted.eq(labels).sum().item()

            total_correct += correct
            total_loss += loss.item() * texts.size(0)
            total_samples += texts.size(0)

            # Print a few sample predictions and their raw text inputs
            if printed_samples < num_samples_to_print:
                for i in range(min(num_samples_to_print - printed_samples, texts.size(0))):
                    true_label = labels[i].item()
                    pred_label = predicted[i].item()
                    print(f"Sample {printed_samples + 1}:")
                    print(f"Raw Text: {raw_texts[i]}")
                    print(f"Predicted: {pred_label}, True: {true_label}")
                    print("-" * 80)
                    printed_samples += 1
                if printed_samples >= num_samples_to_print:
                    break

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return avg_loss, accuracy

# Run evaluation
avg_loss, accuracy = evaluate_model(model, test_loader)

print(f'Test Accuracy: {accuracy:.2f}%')
print(f'Test Average Loss: {avg_loss:.4f}')
