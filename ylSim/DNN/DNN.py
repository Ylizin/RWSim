# -*- coding : utf-8 -*-

import numpy as np
import torch
from torch import nn

class DNN(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.outDim = args.outDim
        self.seqLen = args.seqLen
        self.hiddenDim1 = args.hiddenDim1
        self.hiddenDim2 = args.hiddenDim2
        self.hiddenDim3 = args.hiddenDim3
        self.fc1 = nn.Linear(self.seqLen,self.hiddenDim1)
        self.bn1 = nn.BatchNorm1d(self.hiddenDim1)
        # self.relu = nn.RReLU()
        self.relu = nn.RReLU()
        self.fc2 = nn.Linear(self.hiddenDim1,self.hiddenDim2)
        self.bn2 = nn.BatchNorm1d(self.hiddenDim2)

        self.fc3 = nn.Linear(self.hiddenDim2,self.hiddenDim3)
        self.bn3 = nn.BatchNorm1d(self.hiddenDim3)

        self.out = nn.Linear(self.hiddenDim3,self.outDim)
        self.dnn = nn.Sequential(
            self.fc1,
            self.bn1,
            self.relu,
            self.fc2,
            self.bn2,
            self.relu,
            self.fc3,
            self.bn3,
            self.relu,
            self.out,
        )


    def forward(self,seq):
        #assure seq is 1 dim
        seq.view(-1)
        out = self.dnn(seq)
        return out
        
