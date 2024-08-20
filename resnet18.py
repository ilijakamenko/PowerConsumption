import torch

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import time
import subprocess
import csv
import pandas as pd
import matplotlib.pyplot as plt


device_type = "npu"   #specify platform ['cuda', 'npu']
device_number="7"


device=device_type+":"+device_number



# Specify baseline power consumption (in Watts)
baseline_power = 66.4   # 7W for NVIDIA RTX3050;54W for NVIDIA A100; 67W for NPU 910A
print(f'Baseline Power Consumption: {baseline_power:.2f}W')


if device_type=="npu":
    try:
        import torch_npu 
    except:
        pass

# Define the ResNet-18 architecture
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False, num_classes=10)

    def forward(self, x):
        return self.model(x)

# Prepare dataset
transform = transforms.Compose([
    transforms.Resize(224),  # Resize images to 224x224 to fit ResNet input size
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model, loss, and optimizer
model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to measure power consumption
def measure_power():
    if device_type=="cuda":
        result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
        power_draws = result.stdout.decode('utf-8').strip().split('\n')
        return float(power_draws[int(device_number)])
    elif device_type=="npu":
        result = subprocess.run(['npu-smi','info','-t', 'power','-i', device_number, '-c', '0'], stdout=subprocess.PIPE)
        return float(result.stdout.decode('utf-8').strip().split(":")[1])
    elif device=="cpu":
        return 0 # add logic for cpu
    else:
        return 0  

# Store power measurements for each epoch
def log_power_measurement(start_time, epoch, measurements):
    current_time = time.time()
    elapsed_time = current_time - start_time
    power = round(measure_power() - baseline_power, 2)
    measurements.append((epoch, round(elapsed_time, 2), power))

# Start training
num_epochs = 5
epoch_measurements = {i: [] for i in range(num_epochs)}

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # Log power consumption continuously
        log_power_measurement(epoch_start_time, epoch, epoch_measurements[epoch])
    
    epoch_time = time.time() - epoch_start_time
    print(f'Epoch {epoch+1} completed in {epoch_time:.2f}s')

# Save power measurements to CSV
csv_file = 'power_measurements_'+device_type+'.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Elapsed Time (s)', 'Power Consumption (W)'])
    for epoch, measurements in epoch_measurements.items():
        writer.writerows(measurements)

print(f'Power measurements saved to {csv_file}')


