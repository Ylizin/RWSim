import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn
from CNN.LoadData import concate_narr

_CUDA = torch.cuda.is_available()

ce = nn.NLLLoss(reduction='sum')
# def reconstructed_loss(X_bow,predict_x_bow):
#     deviate = X_bow - predict_x_bow
# def my_weighted_mse(X_bow,predict_x_bow):
#     X_bow_weighted = X_bow + 0.1
#     pow_deviation = (X_bow - predict_x_bow).pow(2)
#     weighted_difference = X_bow_weighted*(pow_deviation)
#     return torch.sum(weighted_difference)#/X_bow.shape[0] #take mean as loss
# mse = my_weighted_mse
cos = nn.CosineSimilarity()

class NTMModel(nn.Module):
    """
    the NTM model
    
    input is a X_bow, its shape is a 1-D (V) tensor, V represent the length of the vocab, 
    and the value of each dimension means the frequency of this word
    
    args.topic_size corresponds to k
    """
    def __init__(self, args,pret = None):
        super().__init__()
        self.pretrained = False
        if not pret is None:
            self.embedding = nn.Embedding.from_pretrained(pret)
            self.pretrained = True

        self.max_length = args.max_length
        self.topic_size = args.topic_size
        self.vocab_size = args.vocab_size
        self.embedding_size= args.embedding_size

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        #this is the matrix of embedding-topic_coefficient
        self.e_t = nn.Linear(self.embedding_size,self.topic_size)
        #this is the matrix of bow-topic 
        self.b_t = nn.Linear(self.vocab_size,self.topic_size)

    def vectorize_bow(self,bow):
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

    def forward(self, embedding , bow):
        bow = self.vectorize_bow(bow)
        
        embedding = self.__tensorize_and_pad(embedding)
        embedding = torch.sum(embedding,dim=1)
        _embedding_t = self.sigmoid(self.relu(self.e_t(embedding))).unsqueeze(2)
        _bow_t = self.softmax(self.relu(self.b_t(bow))).unsqueeze(1)
        _out = torch.bmm(_bow_t,_embedding_t).squeeze(2)# the out put should be (bzs,1,1)
        _out = torch.log(_out)
        return ce(_out,torch.zeros_like(_out.squeeze(1),dtype=torch.long)),_bow_t.squeeze()

