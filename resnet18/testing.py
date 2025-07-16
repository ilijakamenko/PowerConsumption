# ---------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from device_info import DeviceInfo

from results import TestResult

try:
    import torch_npu
except:
    pass
# ---------------------------------------------------------------------------------------------


def evaluate_model(model, test_loader, device: str):
    total_correct = 0
    total_loss = 0
    total_samples = 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)

            _, predicted = torch.max(output, 1)
            correct = predicted.eq(target).sum().item()

            total_correct += correct
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return avg_loss, accuracy


def test(model: nn.Module, device_info: DeviceInfo) -> TestResult:
    model.eval()

    test_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    test_dataset = datasets.MNIST(
        "data", train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )  # for a100 add num_workers>1

    avg_loss, accuracy = evaluate_model(model, test_loader, device_info.device)

    return TestResult(accuracy=accuracy, avg_loss=avg_loss)
