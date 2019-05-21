import numpy as np
import torch
from torch import nn

from .TMNModel import TMNModel
from .NTMModel import NTMModel

class TMN(nn.Module): 
    def __init__(self,args,vae_model):
        super().__init__()
        self.TMN = TMNModel(args,vae_model)
        self.cos = nn.CosineSimilarity()
    
    def fine_tune_parameters(self):
        return self.TMN.fine_tune_parameters()

    def forward(self, bow1,bow2,f1,f2):
        strengthened_f1 = self.TMN(bow1,f1)
        strengthened_f2 = self.TMN(bow2,f2)
        r1 = torch.sum(strengthened_f1,dim=1)
        r2 = torch.sum(strengthened_f2,dim=1)

        return self.cos(r1,r2)*3
    
    