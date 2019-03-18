import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


_CUDA = torch.cuda.is_available()

mse = nn.MSELoss()
# def reconstructed_loss(X_bow,predict_x_bow):
#     deviate = X_bow - predict_x_bow
    


class NTMModel(nn.Module):
    """
    the NTM model
    
    input is a X_bow, its shape is a 1-D (V) tensor, V represent the length of the vocab, 
    and the value of each dimension means the frequency of this word
    
    args.topic_size corresponds to k
    """

    def __init__(self, args):
        super().__init__()

        self.vocab_size = args.vocab_size
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.encoder = nn.Linear(args.vocab_size, args.hidden_size1)
        self.f_mu = nn.Linear(args.hidden_size1, args.topic_size)
        self.f_sigma = nn.Linear(args.hidden_size1, args.topic_size)
        self.f_theta = nn.Linear(args.topic_size, args.topic_size)
        self.f_phi = nn.Linear(args.topic_size, args.vocab_size)

    def reparameterize(self, mu, log_var):
        #std is the standard deviation , is the sigma
        #
        std = log_var.mul(0.5).exp()
        if _CUDA:
            eps = torch.normal(mu,std).cuda()
        else:
            eps = torch.normal(mu,std)
        return eps

    def __vectorize_bow(self,bow):
        len_bow = len(bow)
        stacked_bow = []
        for idx_freq in bow:
            tensor_bow = torch.zeros(self.vocab_size)
            for idx,freq in idx_freq:
                tensor_bow[idx] = freq
            stacked_bow.append(tensor_bow)
        stacked_bow = torch.stack(stacked_bow)
        if _CUDA:
            stacked_bow = stacked_bow.cuda()
       
        return stacked_bow

    def forward(self, X_bow):
        X_bow = self.__vectorize_bow(X_bow)
        pi = self.relu(self.encoder(X_bow))
        mu = self.relu(self.f_mu(pi))
        log_var = self.relu(self.f_sigma(pi))
        z = self.reparameterize(mu, log_var)
        theta = self.relu(self.f_theta(z))
        
        theta = self.softmax(theta)
      
        out_bow = self.relu(self.f_phi(theta))
        return out_bow, theta, mu, log_var, X_bow
        # the loss should be calculated by BCELoss and pass the X_bow as weight

    @staticmethod
    def loss_function(X_bow, predict_x_bow, mu, log_var):
        
        mse_loss = mse(X_bow, predict_x_bow)
        KLD_element = mu.pow(2).add(log_var.exp()).mul(-1).add(1).add(log_var)
        KLD = torch.sum(KLD_element).mul(-0.5)
        return mse_loss + KLD
