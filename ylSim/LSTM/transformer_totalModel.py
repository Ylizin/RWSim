import torch
from torch import nn
from LSTM.TransformerModel import Transformer as tsfm
import LSTM.CalSim as Sim



'''
for transformer training procedure, the GPU memory consumption is too severely 
because of the bp needs store the input
'''

class RWTSMModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.transf= tsfm(args)
        self.bi = Sim.CalSim(args)

    def forward(self, inseq1 , inseq2):
        output1 = self.transf(inseq1)
        output2 = self.transf(inseq2)
        return self.bi(output1,output2)