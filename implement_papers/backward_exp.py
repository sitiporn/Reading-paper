import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray



def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss

model = nn.Linear(2, 2)
x = torch.randn(1, 2)
target = torch.randn(1, 2)
output = model(x)
loss = my_loss(output, target)
print(loss.shape)
print(type(loss))
loss.backward()
print(model.weight.grad)
