#---------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
try:
    import torch_npu 
except:
    pass 
#---------------------------------------------------------------------------------------------


device_type = "cuda"   # specify platform ['cuda', 'npu', 'cpu']
device_number = "0"
device = device_type + ":" + device_number

# Load the model
model_file="models/"+os.path.basename(__file__).split('_')[0]+'_'+device_type+'.pt'
print(f"Loading model from {model_file}")
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False, num_classes=10)

    def forward(self, x):
        return self.model(x)
model = torch.load(model_file, weights_only=False).to(device)
model.eval()  # Set model to evaluation mode

# Define the test dataset transformations (similar to training)
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Prepare the test dataset
test_dataset = datasets.MNIST('data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  #for a100 add num_workers>1

# Define the criterion (loss function)
criterion = nn.CrossEntropyLoss()

# Helper function to unnormalize and display image
def imshow(img, title):
    img = img.cpu().numpy().transpose((1, 2, 0))
    img = np.clip(img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406)), 0, 1)  # Unnormalize
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

# Evaluate the model and plot sample predictions
def evaluate_model_and_plot(model, test_loader, num_samples_to_plot=5):
    total_correct = 0
    total_loss = 0
    total_samples = 0

    # Initialize figure for plotting
    fig, axs = plt.subplots(1, num_samples_to_plot, figsize=(15, 5))
    sample_count = 0

    with torch.no_grad():  # No gradient computation during evaluation
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            # Compute accuracy
            _, predicted = torch.max(output, 1)
            correct = predicted.eq(target).sum().item()

            total_correct += correct
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

            # Plot a few sample predictions
            if sample_count < num_samples_to_plot:
                for i in range(min(num_samples_to_plot - sample_count, data.size(0))):
                    img = data[i]
                    true_label = target[i].item()
                    pred_label = predicted[i].item()
                    title = f'Pred: {pred_label}, True: {true_label}'
                    imshow(img, title)  # Show image with title
                    axs[sample_count].imshow(img.cpu().numpy().transpose(1, 2, 0))
                    axs[sample_count].set_title(title)
                    axs[sample_count].axis('off')
                    sample_count += 1
                if sample_count >= num_samples_to_plot:
                    break

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    plt.tight_layout()
    plt.show()

    return avg_loss, accuracy

# Run evaluation and plot
avg_loss, accuracy = evaluate_model_and_plot(model, test_loader)

print(f'Test Accuracy: {accuracy:.2f}%')
print(f'Test Average Loss: {avg_loss:.4f}')

