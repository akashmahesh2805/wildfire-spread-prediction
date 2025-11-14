import torch
print("is_available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
print("current device:", torch.cuda.current_device() if torch.cuda.is_available() else "None")