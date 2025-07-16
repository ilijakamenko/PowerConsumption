class DeviceInfo:
    def __init__(self, device_type: str, device_number: str):
        self.device = f"{device_type}:{device_number}"
        self.device_number = device_number
        self.device_type = device_type
