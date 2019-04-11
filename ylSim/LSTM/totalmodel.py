import torch
from torch import nn
import LSTM.RWLSTM as RWLSTM
import LSTM.CalSim as Sim
from LSTM.RWLSTM import _CUDA

class RWLSTMModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.lstm = RWLSTM.RWLSTM(args)
        self.bi = Sim.CalSim(args)

    def forward(self, inseq1 , inseq2):
        output1,h1,c1,input1 = self.lstm(inseq1)
        output2,h2,c2,input2 = self.lstm(inseq2)
        return self.bi(output1,output2)

    