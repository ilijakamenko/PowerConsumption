from device_info import DeviceInfo
from lstm.benchmark import LSTMBenchmark
from resnet18.benchmark import Resnet18Benchmark
from transformer.benchmark import TransformerBenchmark


if __name__ == "__main__":
    device_info = DeviceInfo(device_type="cuda", device_number="0")
    lstm_benchmark = LSTMBenchmark(device_info)
    lstm_benchmark.run()

    transformer_benchmark = TransformerBenchmark(device_info)
    transformer_benchmark.run()

    resnet18_benchmark = Resnet18Benchmark(device_info)
    resnet18_benchmark.run()
