import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .NTMModel import _CUDA,cos

class ATTSModel(nn.Module):
    def __init__(self,args,vae_model):
        super().__init__()
        self.vae = vae_model
        #att use the result of cos_sim or the raw embedding or the raw theta?
        self.att = nn.Bilinear(args.topic_size,args.topic_size,args.topic_size,bias=False)
        self.topic_embedding = vae_model.topic_embedding.weight
        self.cosine = cos
        self.softmax = nn.Softmax(dim= 1)

    def forward(self, req_b,wsdl_b):
        _, req_theta, _, _,req_embedding = self.vae(req_b)
        _, wsdl_theta, _, _,wsdl_embedding = self.vae(wsdl_b)

        #topic embedding is topic_num*Embedding_num
        t_topic_embedding = self.topic_embedding
        req_topic_sim = torch.matmul(req_embedding,t_topic_embedding)
        wsdl_topic_sim = torch.matmul(wsdl_embedding,t_topic_embedding)
        att_weight = self.softmax(self.att(req_topic_sim,wsdl_topic_sim))
        #req_theta-wsdl_theta -> N,topic_num , att_weight -> N,topic_size 
        att_dist = torch.sum(torch.abs(req_theta-wsdl_theta) * att_weight, dim = 1)
        return att_dist
    