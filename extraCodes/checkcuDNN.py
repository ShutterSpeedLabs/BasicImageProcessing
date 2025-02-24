import torch
print(torch.backends.cudnn.version())  # Check cuDNN version
print(torch.cuda.is_available())  # Ensure CUDA is available
print(torch.backends.cudnn.enabled)  # Check if cuDNN is enabled
