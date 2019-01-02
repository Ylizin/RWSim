import torch
from torch import nn

class RWBiLinear(nn.Module):
    def __init__(self,args):
        super().__init__()
        if args.bidirectional:
            self.inf = 2*args.hidden_size 
        else:
            self.inf = args.hidden_size 

        self.outf = args.outDim
        self.bi = nn.Bilinear(self.inf,self.inf,self.outf)

    def forward(self,feature1, feature2):
        return self.bi(feature1,feature2)

    
        