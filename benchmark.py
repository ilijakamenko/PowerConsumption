import os
from abc import ABC, abstractmethod

from torch import nn
import torch

from config import BENCHMARKS_DIR, MODELS_DIR
from device_info import DeviceInfo
from results import TestResult, TrainingResults
from tqdm import tqdm
import pandas as pd


def cleanup_cuda():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def save_model(model, model_name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, model_name)
    torch.save(model, model_path)


class Benchmark(ABC):
    def __init__(
        self,
        model_name: str,
        device_info: DeviceInfo,
        num_workers: int = 1,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.device_info = device_info
        self.num_workers = num_workers
        self.batch_size = batch_size

    @abstractmethod
    def train(self) -> TrainingResults:
        pass

    @abstractmethod
    def test(self, model: nn.Module) -> TestResult:
        pass

    def run(
        self,
        run_count: int = 3,
        should_save_model: bool = False,
    ):
        runs = []
        for run in tqdm(range(run_count), desc="Running benchmark"):
            cleanup_cuda()
            # Warmup train, discard results
            tqdm.write("Warming up GPU...")
            self.train()
            torch.cuda.synchronize()
            tqdm.write("Warmed up")

            train_results = self.train()
            cleanup_cuda()
            total_params = sum(p.numel() for p in train_results.model.parameters())
            for m in train_results.measurements:
                runs.append(
                    {
                        "run": run,
                        "epoch": m.epoch,
                        "elapsed_time": m.elapsed_time,
                        "power_consumption": m.power_consumption,
                        "accuracy": m.avg_accuracy,
                        "avg_loss": m.loss,
                        "total_params": total_params,
                        "batch_size": self.batch_size,
                    }
                )

            if should_save_model:
                model_name = f"lstm_{self.device_info.device_type}_{run}.pt"
                save_model(train_results.model, model_name)

        df = pd.DataFrame(runs)
        self.save_run(df)

    def save_run(self, df: pd.DataFrame):
        os.makedirs(BENCHMARKS_DIR, exist_ok=True)
        df.to_csv(
            os.path.join(
                BENCHMARKS_DIR,
                f"{self.model_name}_benchmark_{self.device_info.device_type}.csv",
            ),
            index=False,
        )
