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
        self.cs = nn.CosineSimilarity()
        self.norm = nn.BatchNorm1d(self.inf)

    def forward(self,feature1, feature2):
        feature1 = torch.sum(feature1,1)
        feature2 = torch.sum(feature2,1)
        # feature1 = self.norm(feature1)
        # feature2 = self.norm(feature2)
        # return torch.sum(torch.mul(feature1,feature2),dim = 1)
        return self.cs(feature1,feature2)*3