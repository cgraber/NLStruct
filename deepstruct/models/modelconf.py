import torch

#GLOBALS
USE_GPU = False


if USE_GPU:
    tensor_mod = torch.cuda
else:
    tensor_mod = torch

def use_gpu():
    global USE_GPU, tensor_mod
    USE_GPU = True
    tensor_mod = torch.cuda
