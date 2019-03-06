import torch
from torch import nn
import LSTM.RWBertModel as RWBertModel
import LSTM.RWLSTM as RWLSTM

import LSTM.CalSim as Sim

_CUDA = torch.cuda.is_available()
class BERTTotalModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.bert = RWBertModel.RWBertModel()
        self.lstm = RWLSTM.RWLSTM(args)
        self.bi = Sim.CalSim(args)
        if _CUDA:
            self.lstm = self.lstm.cuda()
            self.bi = self.bi.cuda()

    def forward(self, inseq1 , inseq2):
        CLS,v1,v2 = self.bert(inseq1,inseq2)
        output1,h1,c1,input1 = self.lstm(v1)
        output2,h2,c2,input2 = self.lstm(v2)        
        
        return self.bi(output1,output2)