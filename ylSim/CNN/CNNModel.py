import numpy as np
import torch
from torch import nn

_CUDA = torch.cuda.is_available()


class CNNModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mid_channel = args.mid_channel
        self.in_channel = 1  # args.out_channels
        self.kernel_size = args.kernel_size  # here take a square kernel
        self.max_length = args.max_length  # default length of the input doc

        self.pret = args.pret
        if self.pret:  # if pret is not None then the input will be LongTensors of index
            
            # here we figure out that the padding idx is -1 and during the embedding procedure, idx -1 will be initialized to zero
            self.embedding = nn.Embedding.from_pretrained(pret, padding_idx=-1)

        self.padding = int(
            (self.kernel_size - 1) / 2
        )  # using this padding value can generate the same shape as input
        self.convs = nn.Sequential(
            nn.Conv2d(
                self.in_channel,
                self.mid_channel,
                self.kernel_size,
                padding=self.padding,
            ),
            nn.Conv2d(
                self.mid_channel,
                self.in_channel,
                self.kernel_size,
                padding=self.padding,
            ),
            nn.Tanh(),
        )
        
    def forward(self,in_feature):

        
        if not self.pret:
            #add channel dimension 
            in_feature = in_feature.unsqueeze(1)
            return self.convs(in_feature)
