import torch
import torch_directml

# Check if DirectML is available
if torch_directml.is_available():
    print("DirectML is ALIVE.")
    
    # Assign the device
    device = torch_directml.device()
    print("Device name:", torch_directml.device_name(torch_directml.default_device()))
    
    # Do a test tensor calculation on the GPU
    x = torch.tensor([1.0, 2.0]).to(device)
    y = torch.tensor([3.0, 4.0]).to(device)
    print("GPU Math Test:", x + y)
else:
    print("DirectML failed to load.")