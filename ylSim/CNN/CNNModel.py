import numpy as np
import torch
from torch import nn

_CUDA = torch.cuda.is_available()

class CNNModel(nn.Module):
    def __init__(self,args,pret = None):
        super().__init__()
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.kernel_size = args.kernel_size
        self.max_length = args.max_length #default length of the input doc
        
        self.is_pret = False
        if pret: #if pret is not None then the input will be LongTensors of index
            self.is_pret = True
            self.embedding = nn.Embedding.from_pretrained(pret)
        
        