import torch, torchvision
print(torch.__version__)       # Должно вывести 2.0.1+cu118
print(torchvision.__version__) # Должно вывести что-то вроде 0.15.2+cu118
print(torch.cuda.is_available())  # Должно быть True
