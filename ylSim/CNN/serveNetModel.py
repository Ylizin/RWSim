import numpy as np
import torch
from torch import nn

from .CNNModel import _CUDA,CNNModel

class serveNet(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.CNN = CNNModel(args)
        self.cos = nn.CosineSimilarity()

    def forward(self,inf1,inf2):

        
        feature1 = self.CNN(inf1)
        feature2 = self.CNN(inf2)
        
        sum_f1 = torch.sum(feature1,dim=1)
        sum_f2 = torch.sum(feature2,dim=1)

        return self.cos(sum_f1,sum_f2)