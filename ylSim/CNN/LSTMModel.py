import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import *

class biLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 300
        self.hidden_size = args.hidden_size
        self.rnn = nn.LSTM(self.input_size,self.hidden_size,1,bidirectional = True,batch_first=True)
    
    def forward(self, seqs):
        out,_ = self.rnn(seqs)
        return out
        