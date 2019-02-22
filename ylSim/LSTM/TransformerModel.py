import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import *

from transformer.Layers import EncoderLayer
_CUDA = torch.cuda.is_available()

class Transformer(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.n_layers = 3
        #d_inner is for ff layer, it maps d_model to d_inner and then from d_inner to d_model
        self.encoderLayer = nn.ModuleList([EncoderLayer(d_model = 300, d_inner = 1200,n_head = 6, d_k =50,d_v=50,dropout=0.0) for _ in range(self.n_layers)])
        self.dropout = nn.Dropout(args.dropout)

    def __list_index_select(self,li,idx):
        ret = []
        for i in idx:
            _tensor = torch.from_numpy(li[i])
            if _CUDA :
                _tensor = _tensor.cuda()
            ret.append(_tensor)
        return ret

    def forward(self, seqs):
        curren_batchsize = len(seqs)
        lengths = [k.shape[0] for k in seqs]
        lengths = torch.tensor(lengths)
        if _CUDA:
            lengths = lengths.cuda()

        _,idx_sort = torch.sort(lengths,descending = True)
        _,idx_unsort = torch.sort(idx_sort)
            
        tsm_input = pad_sequence(self.__list_index_select(seqs,idx_sort),batch_first = True)

        raw_padded_input = tsm_input.index_select(0,idx_unsort)
        sum_input = torch.sum(raw_padded_input,1)
        lengths = lengths.type_as(sum_input)
        ave_input = torch.div(sum_input,lengths.view(-1,1))

        output = tsm_input
        
        for layer in self.encoderLayer:
            output,*_ = layer(output)

        sum_output = torch.sum(output,1)
        #here we take the output as the ave of words, lengths been viewed as 128,1 for broadcastable calculation
        lengths = lengths.type_as(sum_output)
        output = torch.div(sum_output,lengths.view(-1,1))

        output = self.dropout(output)

        return output

