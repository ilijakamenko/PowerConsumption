#---------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import subprocess
import csv
import os
try:
    import torch_npu 
except:
    pass
#---------------------------------------------------------------------------------------------

device_type = "cuda"   #specify platform ['cuda', 'npu', 'cpu']
device_number="0"

device=device_type+":"+device_number

# Define the ResNet-18 architecture
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False, num_classes=10)

    def forward(self, x):
        return self.model(x)

# Prepare dataset
transform = transforms.Compose([
    transforms.Resize(224),  # Resize images to 224x2234 to fit ResNet input size
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) #for a100 add num_workers>1

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

# Specify baseline power consumption (in Watts)
baseline_power = round(measure_power(), 2)
print(f'Baseline Power Consumption: {baseline_power:.2f}W')

# Start training
num_epochs = 5
epoch_measurements = {i: [] for i in range(num_epochs)}
print('Start training...')
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
    
# save model
torch.save(model, "models/"+os.path.basename(__file__).split('.')[0]+'_'+device_type+'.pt')

# Save power measurements to CSV
csv_file = os.path.basename(__file__).split(".")[0]+'_power_measurements_'+device_type+'.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Elapsed Time (s)', 'Power Consumption (W)'])
    for epoch, measurements in epoch_measurements.items():
        writer.writerows(measurements)

print(f'Power measurements saved to {csv_file}')


