import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import *

from pytorch_pretrained_bert import BertModel, BertTokenizer

_CUDA = torch.cuda.is_available()
_bert_type = 'bert-base-uncased'

class RWBertModel(nn.Module):
    def __init__(self,type = _bert_type):
        super().__init__() 
        self.tokenizer = BertTokenizer.from_pretrained(_bert_type)
        self.model = BertModel.from_pretrained(_bert_type)
        self.model.eval()
    

    def __vectorize(self,sentence1,sentence2):

        t_s1 = self.tokenizer.tokenize(sentence1)
        t_s2 = self.tokenizer.tokenize(sentence2)

        #record the len of tokenized sentence
        len_s1 = len(t_s1)
        len_s2 = len(t_s2)

        t_s1 = ['[CLS]'] + t_s1 + ['[SEP]']
        t_s2 = t_s2 + ['[SEP]']


        i_s1 = self.tokenizer.convert_tokens_to_ids(t_s1)
        i_s2 = self.tokenizer.convert_tokens_to_ids(t_s2)

        segment_ids = [0 for _ in t_s1]
        segment_ids.extend([1 for _ in t_s2])

        i_sentences = i_s1 + i_s2

        i_tensor = torch.tensor([i_sentences]).view(1,-1)
        s_tensor = torch.tensor([segment_ids]).view(1,-1)

        if torch.cuda.is_available():
            i_tensor = i_tensor.cuda()
            s_tensor = s_tensor.cuda()

        hidden_state,pooled_hidden = self.model(i_tensor,s_tensor,output_all_encoded_layers=False)
  
        hidden_state = hidden_state.squeeze(dim = 0)

        CLS = hidden_state[0]
        v_s1 = hidden_state[1:len_s1+1] 
        v_s2 = hidden_state[len_s1+2:]
        if _CUDA:
            CLS = CLS.cuda()
            v_s1 = v_s1.cuda()
            v_s2 = v_s2.cuda()
        return CLS,v_s1,v_s2
    
    def forward(self,sentences1,sentences2):
        CLS=[]
        v1 = []
        v2 = []
        for s1,s2 in zip(sentences1,sentences2):
            _CLS,_v1,_v2 =self.__vectorize(s1,s2)
            CLS.append(CLS)
            v1.append(_v1)
            v2.append(_v2)

        return CLS,v1,v2