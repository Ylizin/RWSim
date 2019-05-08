import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn

_CUDA = torch.cuda.is_available()

mse = nn.MSELoss(reduction='sum')
# def reconstructed_loss(X_bow,predict_x_bow):
#     deviate = X_bow - predict_x_bow
# def my_weighted_mse(X_bow,predict_x_bow):
#     X_bow_weighted = X_bow + 0.1
#     pow_deviation = (X_bow - predict_x_bow).pow(2)
#     weighted_difference = X_bow_weighted*(pow_deviation)
#     return torch.sum(weighted_difference)#/X_bow.shape[0] #take mean as loss
# mse = my_weighted_mse
cos = nn.CosineSimilarity()

def nnl(x_bow,predict_x_bow):
    return - torch.sum(x_bow * torch.log(predict_x_bow/(x_bow+1e-4)+1e-32))

class NTMModel(nn.Module):
    """
    the NTM model
    
    input is a X_bow, its shape is a 1-D (V) tensor, V represent the length of the vocab, 
    and the value of each dimension means the frequency of this word
    
    args.topic_size corresponds to k
    """
    kl_strength = torch.tensor(1.0)
    def __init__(self, args,pret = None):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.pretrained = args.pretrained
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.word_embedding = nn.Linear(args.vocab_size,args.embedding_size,bias=False)
        self.topic_embedding = nn.Linear(args.topic_size,args.embedding_size,bias=False)
        self.encoder = self.word_embedding
        self.f_mu = nn.Linear(args.embedding_size, args.topic_size)
        self.f_sigma = nn.Linear(args.embedding_size, args.topic_size)
        self.f_theta = nn.Linear(args.topic_size, args.topic_size)

        self.f_phi = nn.Linear(args.topic_size, args.vocab_size)

        if _CUDA:
            NTMModel.kl_strength = NTMModel.kl_strength.cuda()
        if not pret is None:
            self.word_embedding.weight=Parameter(pret)
     

    def reparameterize(self, mu, log_var):
        #std is the standard deviation , is the sigma
        #
        std = log_var.mul(0.5).exp()
        if _CUDA:
            eps = torch.normal(mu,std).cuda()
        else:
            eps = torch.normal(mu,std)
        return eps

    def vectorize_bow(self,bow):
        len_bow = len(bow)
        stacked_bow = []
        if self.pretrained:
            stacked_bow = bow
        else:
            for idx_freq in bow:
                tensor_bow = torch.zeros(self.vocab_size)
                for idx,freq in idx_freq:
                    tensor_bow[idx] = freq
                stacked_bow.append(tensor_bow)
        stacked_bow = torch.stack(stacked_bow)
        
        if _CUDA:
            stacked_bow = stacked_bow.cuda()
        
        return stacked_bow

    def fine_tune_parameters(self):
        w_e_id = list(map(id,self.word_embedding.parameters()))
        
        all_params = self.parameters()
        other_params = filter(lambda x: id(x) not in w_e_id, all_params)
        w_e = self.word_embedding.parameters()

        return [{'params':other_params},{'params':list(w_e),'lr':3e-4}]

    def forward(self, X_bow):
        X_bow = self.vectorize_bow(X_bow)
        pi = self.relu(X_bow)
        word_embedding = torch.clone(pi)
        if not self.pretrained:
            word_embedding = self.word_embedding(X_bow)
            _X_bow = self.encoder(X_bow)
            pi = self.relu(_X_bow)
        
        mu = self.relu(self.f_mu(pi))
        log_var = self.relu(self.f_sigma(pi))
        z = self.reparameterize(mu, log_var)
        theta = self.relu(self.f_theta(z))
        theta = self.softmax(theta)
        # out_bow = None
        out_bow = self.relu(self.f_phi(theta))
        # X_bow = word_embedding
        # if not self.pretrained:
        #     out_bow = self.relu(self.f_phi(theta))
        # else:
        #     topic_embedding = self.relu(self.topic_embedding(theta))
        #     out_bow = topic_embedding
        #     X_bow = word_embedding
        return out_bow, theta, mu, log_var, X_bow
        # the loss should be calculated by BCELoss and pass the X_bow as weight

    @staticmethod
    def loss_function(X_bow, predict_x_bow, mu, log_var):
        #X_bow & predict_x_bow is batch*vocab_size, mu and log_var is the same
        # mse_loss = torch.sum(1-cos(X_bow,predict_x_bow))
        mse_loss = nnl(X_bow, predict_x_bow)
        KLD_element = mu.pow(2).add(log_var.exp()).mul(-1).add(1).add(log_var)
        KLD = torch.sum(KLD_element).mul(-0.5)
        
        return mse_loss + KLD * NTMModel.kl_strength
