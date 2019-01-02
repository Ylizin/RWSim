import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import *

class  RWLSTM(nn.Module):
    def __init__(self,args):
        super().__init__()
        #input size : the length of input x, here is the dimension of wordembeddings
        #hidden_size : the size of hidden states, including h and c
        self.input_size = args.input_size
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.droprate = args.dropout
        self.dropout = nn.Dropout(self.droprate)
        self.rnn = nn.LSTM(self.input_size,self.hidden_size,bidirectional = args.bidirectional)

    def __list_index_select(self,li,idx):
        ret = []
        for i in idx:
            ret.append(torch.from_numpy(li[i]))
        return ret

    def __init_hidden(self):
        #hidden state should be (dirctions*layers,batch,hidden)
        return torch.randn(2,self.batch_size,self.hidden_size),torch.randn(2,self.batch_size,self.hidden_size)

    def forward(self, seqs):
        '''
            seqs is an unordered list of ndarray
            here we gonna do some process on the input seq then feed it into LSTM
            first: we do sort on the input with the key of len of each sentence
            second: through torch...pack_sequence, we can get the input of LSTM as a packed seq
            thirdly: we unsort the output of the LSTM then return it for sim calculation afterwards
        '''
        lengths = [k.shape[0] for k in seqs]
        lengths = torch.tensor(lengths)
        #idx_sort is the sorted idx by lengths, which is applied in generating the lstm-input
        #idx_unsort is the unsorted idx, which is leveraged to restore the input
        _,idx_sort = torch.sort(lengths,descending = True)
        _,idx_unsort = torch.sort(idx_sort)
        lstm_input = pack_sequence(self.__list_index_select(seqs,idx_sort))
        h0,c0 = self.__init_hidden()
        lstm_output,(hn,cn) = self.rnn(lstm_input,(h0,c0))
        lstm_output = self.dropout(lstm_output.data)
        #here the output is packedsequence and after padding, it's b X len X 2*hidden
        #then we do index_select to restore the order of the batch
        output = pad_packed_sequence(lstm_output,batch_first=True)

        # output is padded output_seq of lstm, and restored by the idx 
        # output : (batch, seq_len, num_directions * hidden_size)
        # hn : (num_layers * num_directions, batch, hidden_size)
        # cn : (num_layers * num_directions, batch, hidden_size)
        output = output.index_select(idx_unsort)
        hn = hn.index_select(idx_unsort)
        cn = cn.index_select(idx_unsort)

        return output,hn,cn

        
        
        
