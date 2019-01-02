import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import *

_CUDA = torch.cuda.is_available()

class  RWLSTM(nn.Module):
    def __init__(self,args):
        super().__init__()
        #input size : the length of input x, here is the dimension of wordembeddings
        #hidden_size : the size of hidden states, including h and c
        self.input_size = args.input_size
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.droprate = args.dropout
        self.bidirectional = args.bidirectional
        self.dropout = nn.Dropout(self.droprate)
        self.rnn = nn.LSTM(self.input_size,self.hidden_size,bidirectional = args.bidirectional)

    def __list_index_select(self,li,idx):
        ret = []
        for i in idx:
            _tensor = torch.from_numpy(li[i])
            if _CUDA :
                _tensor = _tensor.cuda()
            ret.append(_tensor)
        return ret

    def __init_hidden(self,current_Batchsize):
        device = None
        if _CUDA :
            device = torch.device('cuda')
        #hidden state should be (dirctions*layers,batch,hidden)
        num_dir = 1
        if self.bidirectional:
            num_dir = 2
        return torch.randn(num_dir,current_Batchsize,self.hidden_size,device = device),torch.randn(num_dir,current_Batchsize,self.hidden_size,device = device)

    def forward(self, seqs):
        '''
            seqs is an unordered list of ndarray
            here we gonna do some process on the input seq then feed it into LSTM
            first: we do sort on the input with the key of len of each sentence
            second: through torch...pack_sequence, we can get the input of LSTM as a packed seq
            thirdly: we unsort the output of the LSTM then return it for sim calculation afterwards
        '''
        curren_batchsize = len(seqs)
        lengths = [k.shape[0] for k in seqs]
        lengths = torch.tensor(lengths)
        if _CUDA:
            lengths = lengths.cuda()
        #idx_sort is the sorted idx by lengths, which is applied in generating the lstm-input
        #idx_unsort is the unsorted idx, which is leveraged to restore the input
        _,idx_sort = torch.sort(lengths,descending = True)
        _,idx_unsort = torch.sort(idx_sort)
            
        lstm_input = pack_sequence(self.__list_index_select(seqs,idx_sort))
        h0,c0 = self.__init_hidden(curren_batchsize)
        lstm_output,(hn,cn) = self.rnn(lstm_input,(h0,c0))
        # lstm_output = PackedSequence(self.dropout(lstm_output.data),lstm_output.batch_sizes)

        #here the output is packedsequence and after padding, it's b X len X 2*hidden
        #then we do index_select to restore the order of the batch
        #this pad func return a tuple of (tensor,length)
        output = pad_packed_sequence(lstm_output,batch_first=True)[0]

        # output is padded output_seq of lstm, and restored by the idx 
        # output : (batch, seq_len, num_directions * hidden_size)
        # hn : (num_layers * num_directions, batch, hidden_size)
        # cn : (num_layers * num_directions, batch, hidden_size)
        output = output.index_select(0,idx_unsort) #dim,index
        hn = hn.index_select(1,idx_unsort)
        cn = cn.index_select(1,idx_unsort)
        
        hn = hn.view(hn.size()[1],-1)
        cn = cn.view(cn.size()[1],-1)
        hn = self.dropout(hn)
        cn = self.dropout(cn)
        return output,hn,cn