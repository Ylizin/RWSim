import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .NTMModel import _CUDA,cos

class ATTSModel(nn.Module):
    def __init__(self,args,vae_model):
        super().__init__()
        self.vae = vae_model
        #bi use the result of cos_sim or the raw embedding or the raw theta?
        self.bi = nn.Bilinear(args.topic_size,args.topic_size,args.topic_size,bias=False)
        self.topic_embedding = vae_model.topic_embedding.weight
        self.cosine = cos
        self.softmax = nn.Softmax(dim= 1)

    def forward(self, req_b,wsdl_b):
        _, req_theta, _, _,req_embedding = self.vae(req_b)
        _, wsdl_theta, _, _,wsdl_embedding = self.vae(wsdl_b)
        req_theta = self.softmax(req_theta)
        wsdl_theta = self.softmax(wsdl_theta)

        topic_embedding = self.topic_embedding
        t_topic_embedding = topic_embedding.t()
        

        #here we do att part
        req_embedding = torch.mul(req_theta.unsqueeze(-1),t_topic_embedding) #here we expand the raw_foc into a topic_num*embedding doc 
        wsdl_embedding = torch.mul(wsdl_theta.unsqueeze(-1), t_topic_embedding) # in order to conduct a process further
        att_matrix = torch.bmm(req_embedding,wsdl_embedding.permute(0,2,1))#the result will be batch*len_req*len_wsdl
        att_matrix = self.softmax(att_matrix) # the second dimension is normalized, sum to 1
        att_matrix = att_matrix.permute(0,2,1) # att is permute to be batch*len_wsdl*len_req
        wsdl_embedding = torch.bmm(att_matrix,req_embedding) # this is gonna be batch*len_wsdl*embedding
        ave_req = req_embedding.mean(dim = 1)
        ave_wsdl = wsdl_embedding.mean(dim = 1)
        bi_dist = self.cosine(ave_req,ave_wsdl) * 3

        # #topic embedding is topic_num*Embedding_num
        # req_topic_sim = torch.matmul(req_embedding,topic_embedding)
        # wsdl_topic_sim = torch.matmul(wsdl_embedding,topic_embedding)
        # bi_weight = self.bi(req_topic_sim,wsdl_topic_sim)

        # req_theta = req_theta * bi_weight
        # wsdl_theta = wsdl_theta * bi_weight
        # bi_dist = self.cosine(req_theta,wsdl_theta)*3
        # #req_theta-wsdl_theta -> N,topic_num , bi_weight -> N,topic_size 
        # # bi_dist = torch.sum(torch.abs(req_theta-wsdl_theta) * bi_weight, dim = 1)
        return bi_dist
    