from torch import nn
from benchmark import Benchmark
from device_info import DeviceInfo
from resnet18.testing import test
from resnet18.training import train
from results import TestResult, TrainingResults


class Resnet18Benchmark(Benchmark):
    def __init__(self, device_info: DeviceInfo):
        super().__init__("resnet18", device_info)

    def train(self) -> TrainingResults:
        return train(self.device_info)

    def test(self, model: nn.Module) -> TestResult:
        return test(model, self.device_info)
