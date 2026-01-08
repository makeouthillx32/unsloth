import signal
import torch


print(f"has setitimer: {hasattr(signal, "setitimer")}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.is_available())