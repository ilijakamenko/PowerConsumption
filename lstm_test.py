#---------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
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

# Define the LSTM model (same as the one used during training)
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        hidden = hidden[-1]  # Use the hidden state of the last layer
        output = self.fc(self.dropout(hidden))
        return output

# Load the trained model
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
