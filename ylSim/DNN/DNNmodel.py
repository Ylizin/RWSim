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
        self.droprate = args.drop

        self.drop = nn.Dropout(self.droprate)
        self.fc1 = nn.Linear(self.seqLen,self.hiddenDim1)
        self.bn1 = nn.BatchNorm1d(self.hiddenDim1)
        # self.relu = nn.RReLU()
        self.relu = nn.RReLU()
        self.fc2 = nn.Linear(self.hiddenDim1,self.hiddenDim2)
        self.bn2 = nn.BatchNorm1d(self.hiddenDim2)

        self.fc3 = nn.Linear(self.hiddenDim2,self.hiddenDim3)
        self.bn3 = nn.BatchNorm1d(self.hiddenDim3)

        self.out = nn.Linear(self.hiddenDim3,self.outDim)
        self.dnn1 = nn.Sequential(
            self.fc1,
            # self.bn1,
            self.relu,
            self.fc2,
            self.drop,
            # self.bn2,
            self.relu,
            self.fc3,
            # self.bn3,
            self.relu,
            self.out,
        )
        self.dnn2 = nn.Sequential(
            self.fc1,
            # self.bn1,
            self.relu,
            self.fc2,
            self.drop,
            # self.bn2,
            self.relu,
            self.fc3,
            # self.bn3,
            self.relu,
            self.out,
        )
        self.cs = nn.CosineSimilarity()



    def forward(self,seq1,seq2):
        #assure seq is 1 dim
        seq1.view(-1)
        seq2.view(-1)
        f1 = self.dnn1(seq1)
        f2 = self.dnn2(seq2)
        return self.cs(f1,f2)*3
