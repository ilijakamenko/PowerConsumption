from torch import nn
from benchmark import Benchmark
from device_info import DeviceInfo
from results import TestResult, TrainingResults
from transformer.training import train
from transformer.testing import test


class TransformerBenchmark(Benchmark):
    def __init__(self, device_info: DeviceInfo):
        super().__init__("transformer", device_info)

    def train(self) -> TrainingResults:
        return train(self.device_info)

    def test(self, model: nn.Module) -> TestResult:
        return test(model, self.device_info)
