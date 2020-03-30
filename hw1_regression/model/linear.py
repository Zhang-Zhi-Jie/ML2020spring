import torch
import torch.nn as nn
import numpy as numpy

class LinearNet(nn.Module):
    
    def __init__(self):
        super(LinearNet, self).__init__()
    
        self.linear = nn.Sequential(nn.Linear(162,1))
    
    def forward(self, X):   
        y = self.linear(X)
        return y
