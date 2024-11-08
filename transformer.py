import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import time
import subprocess
import csv
import os
import urllib.request
import tarfile
import math

try:
    import torch_npu
except:
    pass

#---------------------------------------------------------------------------------------------
device_type = "cuda"  # specify platform ['cuda', 'npu', 'cpu']
device_number = "0"
device = device_type + ":" + device_number

# Tokenizer and Vocabulary for IMDB dataset
tokenizer = get_tokenizer('basic_english')

# Function to download the IMDB dataset
def download_imdb_dataset(data_folder='./data/IMDB'):
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    file_path = os.path.join(data_folder, 'aclImdb_v1.tar.gz')
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    # Download the dataset if it doesn't already exist
    if not os.path.exists(file_path):
        print('Downloading IMDB dataset...')
        urllib.request.urlretrieve(url, file_path)
        print('Download completed.')
    
    # Extract the dataset
    print('Extracting dataset...')
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=data_folder)
    print('Extraction completed.')

# Custom Dataset Class
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
        return label, text

def yield_tokens(texts):
    for text in texts:
        yield tokenizer(text)

# Load IMDB dataset from local files
def load_imdb_data(data_folder='./data/IMDB', type='train'):
    texts = []
    labels = []
    
    # Load training data
    for label in ['pos', 'neg']:
        folder = os.path.join(data_folder, type, label)
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1 if label == 'pos' else 0)
    return texts, labels

# Download and extract IMDB dataset
download_imdb_dataset()

# Load data from the extracted IMDB folder
train_texts, train_labels = load_imdb_data('./data/IMDB/aclImdb')

# Build vocabulary
vocab = build_vocab_from_iterator(yield_tokens(train_texts), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Function to preprocess data
def text_pipeline(text):
    return vocab(tokenizer(text))

def label_pipeline(label):
    return 1 if label == "pos" else 0

# Collate function to pad sequences
def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(torch.tensor(label_pipeline(_label)))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
    return torch.tensor(label_list), pad_sequence(text_list, batch_first=True)

# Split data into train and test sets
train_data = IMDBDataset(train_texts, train_labels, vocab)
# DataLoader
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_batch)  # for a100 add num_workers > 1

# Transformer Model Definition
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

# Initialize model, loss, and optimizer
vocab_size = len(vocab)
embed_size = 128
num_heads = 8
hidden_dim = 256
num_layers = 2
output_size = 2  # Binary classification (positive/negative)

model = TransformerModel(vocab_size, embed_size, num_heads, hidden_dim, num_layers, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to measure power consumption
def measure_power():
    if device_type == "cuda":
        result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
        power_draws = result.stdout.decode('utf-8').strip().split('\n')
        return float(power_draws[int(device_number)])
    elif device_type == "npu":
        result = subprocess.run(['npu-smi', 'info', '-t', 'power', '-i', device_number, '-c', '0'], stdout=subprocess.PIPE)
        return float(result.stdout.decode('utf-8').strip().split(":")[1])
    elif device == "cpu":
        return 0  # Add logic for CPU if needed
    else:
        return 0

# Store power measurements for each epoch
def log_power_measurement(start_time, epoch, measurements):
    current_time = time.time()
    elapsed_time = current_time - start_time
    power = round(measure_power() - baseline_power, 2)
    measurements.append((epoch, round(elapsed_time, 2), power))

# Specify baseline power consumption (in Watts)
baseline_power = round(measure_power(), 2)
print(f'Baseline Power Consumption: {baseline_power:.2f}W')

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters in TransformerModel: {total_params}")

# Start training
num_epochs = 5
epoch_measurements = {i: [] for i in range(num_epochs)}
print('Start training...')
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    
    for batch_idx, (label, text) in enumerate(train_loader):
        text, label = text.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        # Log power consumption continuously
        log_power_measurement(epoch_start_time, epoch, epoch_measurements[epoch])
    
    epoch_time = time.time() - epoch_start_time
    print(f'Epoch {epoch + 1} completed in {epoch_time:.2f}s')

# Save the model
torch.save(model, "models/" + os.path.basename(__file__).split('.')[0] + '_' + device_type + '.pt')

# Save power measurements to CSV
csv_file = os.path.basename(__file__).split(".")[0] + '_power_measurements_' + device_type + '.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Elapsed Time (s)', 'Power Consumption (W)'])
    for epoch, measurements in epoch_measurements.items():
        writer.writerows(measurements)

print(f'Power measurements saved to {csv_file}')
