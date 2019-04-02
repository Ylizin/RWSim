import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .NTMModel import _CUDA,cos,NTMModel

class ATTSModel(nn.Module):
    def __init__(self,args,vae_model):
        super().__init__()
        self.vae = vae_model
        self.pretrained = args.pretrained
        #bi use the result of cos_sim or the raw embedding or the raw theta?
        self.bi = nn.Bilinear(args.topic_size,args.topic_size,args.topic_size,bias=False)
        self.topic_embedding = vae_model.topic_embedding.weight
        self.word_embedding = vae_model.word_embedding
        self.cosine = cos
        self.softmax = nn.Softmax(dim= 1)
        self.f_att1 = nn.Linear(args.embedding_size,args.topic_size)
        self.f_att2 = nn.Linear(args.embedding_size,args.topic_size)
        
        self.vae_loss = NTMModel.loss_function

    def fine_tune_parameters(self):
        vae_params_id = list(map(id,self.vae.parameters()))
        
        all_params = self.parameters()
        ntm_params = filter(lambda x:id(x) not in vae_params_id,all_params)
        vae_params = self.vae.parameters()
    
        return [{'params':ntm_params},{'params':vae_params,'lr':3e-5}]


    def forward(self, req_b,wsdl_b):
        #here we do the cosine loss for the word_embedding 
        req_p_bow, req_theta, req_mu, req_var,req_embedding = self.vae(req_b)
        wsdl_p_bow, wsdl_theta, wsdl_mu, wsdl_var,wsdl_embedding = self.vae(wsdl_b)
        # this loss restricts the vae
        # if not self.pretrained:
        #     req_embedding = self.word_embedding(req_p_bow)
        #     wsdl_embedding = self.word_embedding(wsdl_p_bow)
        topic_embedding = self.topic_embedding
        t_topic_embedding = topic_embedding.t()
        
       

        # #topic embedding is topic_num*Embedding_num
        req_topic_sim = torch.matmul(req_embedding,topic_embedding)
        wsdl_topic_sim = torch.matmul(wsdl_embedding,topic_embedding)
        bi_weight = self.bi(req_topic_sim,wsdl_topic_sim)
        # bi_weight = self.softmax(bi_weight)
        # req_theta = req_theta * att_matrix
        # wsdl_theta = wsdl_theta * att_matrix
        # req_theta = req_theta * bi_weight
        # wsdl_theta = wsdl_theta * bi_weight
        # bi_dist = self.cosine(req_theta,wsdl_theta)*3
        # #req_theta-wsdl_theta -> N,topic_num , bi_weight -> N,topic_size 
        bi_dist = torch.sum(torch.abs(req_theta-wsdl_theta) * bi_weight, dim = 1)
        w_e_dist = self.cosine(req_embedding,wsdl_embedding)*3
        return (bi_dist+w_e_dist)/2
    