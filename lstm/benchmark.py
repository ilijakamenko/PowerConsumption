from torch import nn
from benchmark import Benchmark
from device_info import DeviceInfo
from lstm.testing import test
from lstm.training import train
from results import TestResult, TrainingResults


class LSTMBenchmark(Benchmark):
    def __init__(self, device_info: DeviceInfo):
        super().__init__("lstm", device_info)

    def train(self) -> TrainingResults:
        return train(self.device_info)

    def test(self, model: nn.Module) -> TestResult:
        return test(model, self.device_info)
