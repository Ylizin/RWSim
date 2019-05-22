import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn
from .NTMModel import _CUDA, cos, NTMModel
from CNN.LoadData import concate_narr


#code for TMN actually is an attention mechanism, 
# input is L*embedding_size and K*vocab_size
# K*vocab_size is the topic-word matrix
# after the attention we will get a L*K matrix and then convert it to L*embedding_size 
#then purely add it to the original matrix
class TMNModel(nn.Module):
    def __init__(self,args,vae_model):
        super().__init__()
        self.vae = vae_model
        self.vocab_size = args.vocab_size
        self.embedding_size = args.embedding_size
        self.topic_size = args.topic_size
        self.topic_embedding_size = args.topic_embedding_size
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        #f_phi is a topic_size*vocab_size and itcorresponds to the topic-word matrix
        self.topic_embedding = self.vae.f_phi.weight
        self.topic_embedding.requires_grad = False
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        self.rnn = nn.LSTM(self.embedding_size,int(self.embedding_size/2),1,bidirectional = True,batch_first=True)
        self.w = nn.Linear(self.topic_size,self.embedding_size)
        self.u = nn.Linear(self.embedding_size,self.embedding_size)
        self.tanh = nn.Tanh()
        self.v = nn.Linear(self.embedding_size,1)

    def fine_tune_parameters(self):
        vae_params_id = list(map(id, self.vae.parameters()))
        all_params = self.parameters()
        ntm_params = filter(lambda x: id(x) not in vae_params_id, all_params)
        vae_params = self.vae.parameters()

        return [{"params": ntm_params}, {"params": vae_params, "lr": 3e-5}]


    def __tensorize_and_pad(self,input_vectors):
        max_length = self.max_length
        paded = []
        for vec in input_vectors:
            vec = concate_narr(vec,max_length)
            paded.append(vec)
        
        ret = torch.tensor(paded).to(torch.float)
        if _CUDA: 
            ret = ret.cuda()
        return ret
    
    def forward(self, bow_input,feature_input):
        self.batch_size = len(bow_input)
        # the bow will be pass directly into vae
        feature_input = self.relu(self.__tensorize_and_pad(feature_input))
        out_bow,theta,*_ = self.vae(bow_input)
        #theta is (bzs,k), out is (bzs,L,embedding_size)
        out,_ = self.rnn(feature_input)
        _w_theta = self.w(theta)
        _u_h = self.u(out)
        _g = self.softmax(self.v(self.tanh(_w_theta+_u_h)).squeeze()) #this would be (bzs,L,1)

        out = torch.sum(out*_g,dim=1) #(bzs,embedding_size)
        return out
        
        



        