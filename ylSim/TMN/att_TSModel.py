import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn
from .NTMModel import _CUDA, cos, NTMModel


class ATTSModel(nn.Module):
    def __init__(self, args, vae_model):
        super().__init__()
        self.vae = vae_model
        self.pretrained = args.pretrained
        # bi use the result of cos_sim or the raw embedding or the raw theta?
        self.bi = nn.Bilinear(
            args.topic_size, args.topic_size, args.topic_size, bias=False
        )
        self.topic_embedding = vae_model.f_phi.weight
        self.word_embedding = vae_model.word_embedding
        self.cosine = cos
        self.softmax = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()
        self.f_att = nn.Linear(args.embedding_size+args.vocab_size, 1)
        # self.f_out = nn.Linear(2*args.topic_size,1)
        # self.we_out = nn.Linear(2*args.embedding_size,1) 
        self.embedding_size = args.embedding_size
    
        self.topic_size = args.topic_size
        self.vae_loss = NTMModel.loss_function
        self.alpha = Parameter(torch.tensor(1.0))

    def fine_tune_parameters(self):
        vae_params_id = list(map(id, self.vae.parameters()))

        all_params = self.parameters()
        ntm_params = filter(lambda x: id(x) not in vae_params_id, all_params)
        vae_params = self.vae.parameters()

        return [{"params": ntm_params}, {"params": vae_params, "lr": 3e-5}]

    def forward(self, req_b, wsdl_b):
        self.bzs = len(req_b)
        # here we do the cosine loss for the word_embedding
        req_p_bow, req_theta, req_mu, req_var, req_embedding = self.vae(req_b)
        wsdl_p_bow, wsdl_theta, wsdl_mu, wsdl_var, wsdl_embedding = self.vae(wsdl_b)
        # this loss restricts the vae
        # if not self.pretrained:
        #     req_embedding = self.word_embedding(req_p_bow)
        #     wsdl_embedding = self.word_embedding(wsdl_p_bow)
        topic_embedding = self.topic_embedding
        t_topic_embedding = topic_embedding.t()
        _t_topic_embedding = t_topic_embedding.expand(self.bzs,-1,-1)#bzs,120,300
        #region of topic-dw att
        _req_embedding = req_embedding.expand(self.topic_size,-1,-1).permute(1,0,2) #expand do not alloc memory
        _wsdl_embedding = wsdl_embedding.expand(self.topic_size,-1,-1).permute(1,0,2) 
        _req_t = torch.cat([_req_embedding,_t_topic_embedding],dim=-1)
        _wsdl_t = torch.cat([_wsdl_embedding,_t_topic_embedding],dim=-1)
        _req_att = self.f_att(_req_t).squeeze() #bzs,topic_num
        _wsdl_att = self.f_att(_wsdl_t).squeeze() #bzs,topic_num

        # #topic embedding is topic_num*Embedding_num
        # req_topic_sim = torch.matmul(req_embedding, topic_embedding)
        # wsdl_topic_sim = torch.matmul(wsdl_embedding, topic_embedding)
        # bi_weight = self.bi(req_topic_sim, wsdl_topic_sim)
        # bi_weight = self.softmax(bi_weight)
        # req_theta = req_theta * att_matrix
        # wsdl_theta = wsdl_theta * att_matrix
        req_theta = req_theta * _req_att
        wsdl_theta = wsdl_theta * _wsdl_att
    
        # bi_dist = self.f_out(torch.cat([req_theta,wsdl_theta],dim=1)).squeeze()
        # req_topic = torch.matmul(req_theta,t_topic_embedding)
        # wsdl_topic = torch.matmul(wsdl_theta,t_topic_embedding)
        bi_dist = self.cosine(req_theta,wsdl_theta)
        # bi_dist = torch.sum(torch.abs(req_theta - wsdl_theta), dim=1)
   
        #req_theta-wsdl_theta -> N,topic_num , bi_weight -> N,topic_size
        w_e_dist = (self.cosine(req_embedding, wsdl_embedding))
        return (self.alpha*bi_dist + w_e_dist) 
        # return bi_dist

