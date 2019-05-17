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

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        #convert the len*word_embeddings to len*topic_embedding_size
        self.c1 = nn.Linear(self.embedding_size,self.topic_embedding_size)
        #conver topic*vocab to topic*topic_embedding_size
        self.t1 = nn.Linear(self.vocab_size,self.embedding_size)
        #convert the length_of_sen*topic_size to length_of_sen*topic_embedding_size
        self.f1 = nn.Linear(self.topic_size,self.topic_embedding_size)
        #process the total result??
        self.o1 = nn.Linear(self.topic_embedding_size,self.topic_embedding_size)

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
        feature_input = self.relu(self.c1(self.__tensorize_and_pad(feature_input)))
        out_bow,theta,*_ = self.vae(bow_input)

        wt_embedding = self.relu(self.t1(self.topic_embedding.t().expand(self.batch_size,-1,-1)))
        match = torch.bmm(feature_input,wt_embedding.transpose(1,2))# match will be (bz,L,K)
        joint_match = torch.add(theta.expand(self.max_length,-1,-1).transpose(0,1),match)
        joint_match = self.relu(self.f1(joint_match))# (bz,L,topic_embedding_size)
        _feature_strengthed = torch.add(feature_input,joint_match)

        _feature_strengthed=self.relu(self.o1(_feature_strengthed))
        return _feature_strengthed
        
        



        