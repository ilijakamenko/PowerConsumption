from torch import nn
from benchmark import Benchmark
from device_info import DeviceInfo
from results import TestResult, TrainingResults
from transformer.training import train
from transformer.testing import test


class TransformerBenchmark(Benchmark):
    def __init__(
        self,
        device_info: DeviceInfo,
        **kwargs,
    ):
        super().__init__(
            model_name="transformer",
            device_info=device_info,
            **kwargs,
        )

    def train(self) -> TrainingResults:
        return train(
            device_info=self.device_info,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def test(self, model: nn.Module) -> TestResult:
        return test(
            model=model,
            device_info=self.device_info,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )
