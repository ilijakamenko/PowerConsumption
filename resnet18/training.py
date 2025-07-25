# ---------------------------------------------------------------------------------------------
import torch.nn as nn
import torch.optim as optim
from device_info import DeviceInfo
from power import measure_power
from resnet18.model import ResNet18
from results import PowerMeasurement, TrainingResults
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

try:
    import torch_npu
except:
    pass


def train(device_info: DeviceInfo) -> TrainingResults:
    transform = transforms.Compose(
        [
            transforms.Resize(
                224
            ),  # Resize images to 224x2234 to fit ResNet input size
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )  # for a100 add num_workers>1

    model = ResNet18().to(device_info.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    baseline_power = round(measure_power(device_info), 2)

    num_epochs = 5
    measurements = []
    for epoch in tqdm(range(num_epochs), desc="Training"):
        epoch_start_time = time.time()

        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device_info.device), target.to(device_info.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
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
