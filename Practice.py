import torch
import numpy as np
import pandas as pd


x = torch.tensor(2, requires_grad=True)

y=x**2
y.backward()
x.grad
