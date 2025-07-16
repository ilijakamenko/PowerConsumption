# Function to measure power consumption
import subprocess
from device_info import DeviceInfo


def measure_power(device_info: DeviceInfo):
    if device_info.device_type == "cuda":
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
        )
        power_draws = result.stdout.decode("utf-8").strip().split("\n")
        return float(power_draws[int(device_info.device_number)])
    elif device_info.device_type == "npu":
        result = subprocess.run(
            [
                "npu-smi",
                "info",
                "-t",
                "power",
                "-i",
                device_info.device_number,
                "-c",
                "0",
            ],
            stdout=subprocess.PIPE,
        )
        return float(result.stdout.decode("utf-8").strip().split(":")[1])
    elif device_info.device == "cpu":
        return 0  # Add logic for CPU if needed
    else:
        return 0
