import numpy as np
import torch
from torch import nn

from .CNNModel import _CUDA,CNNModel
from .LSTMModel import biLSTM

class serveNet(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.CNN = CNNModel(args)
        self.LSTM = biLSTM(args)
        self.cos = nn.CosineSimilarity()

        if args.with_LSTM:
            self.feature_extract = nn.Sequential(self.CNN,self.LSTM)
        else : 
            self.feature_extract = self.CNN

    def forward(self,inf1,inf2):

        feature1 = self.feature_extract(inf1)
        feature2 = self.feature_extract(inf2)
        

        sum_f1 = torch.sum(feature1,dim=1)
        sum_f2 = torch.sum(feature2,dim=1)

        return self.cos(sum_f1,sum_f2)