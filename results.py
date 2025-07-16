from dataclasses import dataclass
from typing import List
import torch.nn as nn


@dataclass
class PowerMeasurement:
    epoch: int
    elapsed_time: float
    power_consumption: float


@dataclass
class TrainingResults:
    model: nn.Module
    measurements: List[PowerMeasurement]


@dataclass
class TestResult:
    accuracy: float
    avg_loss: float
