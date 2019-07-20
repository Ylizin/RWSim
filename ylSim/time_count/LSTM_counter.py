import numpy as np
import torch
from torch.nn.functional import cosine_similarity as cos
from timeDecorator import time_counter
from utils import dw
from vectorize import word2Vec


def vectorize_words(w2v_model,seqs):
    vecData = None
    for word in seqs:
        wordVec = word2Vec(word)
        if wordVec is None: #if this word is oov,continue
            continue
        if vecData is None:
            vecData = word2Vec(word).reshape(1, dw)
        else:
            vecData = np.append(vecData, word2Vec(word).reshape(1, dw), axis=0)
    return vecData
    

@time_counter
def LSTM_count(w2v_model,lstm_model,input_seqs,ave_registered_tensor,wsdl_names,ret_topk = 20):
    '''this function do the recommendation for the input_seqs
    
    Arguments:
        lstm_model {a pytorch model/callable object} -- [this model is pre-loaded into cuda if possible]
        input_seqs {list of list} -- [a list of list of words which were produced by processing the input wsdls]
        ave_registered_wsdl {a tensor of len_registered_wsdl*dw} -- [its an tensor contains the whole registered wsdls' ave_representation, the orered is 
        correspond to the wsdl_names]
        wsdl_names {tuple} -- [keeps the ordered name of wsdls]
    ''' 
    vec_words = [] #vec_words is a list of ndarray 
    for words in input_seqs:
        vec_words.append(vectorize_words(w2v_model,words))
    ret_tensor,*_ = lstm_model(vec_words)
    # scores = ret_tensor.matmul(ret_tensor,ave_registered_tensor)
    rec_seqs = []
    for tensor in ret_tensor:
        scores = cos(tensor.view(1,-1),ave_registered_tensor)
        values,indices=torch.topk(scores,ret_topk) #topk 实际上支持多维的,cos也支持广播运算
        rec_wsdls = list(wsdl_names[i] for i in indices)
        rec_seqs.append(rec_wsdls)
    return rec_seqs

@time_counter
def VAE_counter(w2v_model,TMN_model,input_seqs,input_tfs,wsdl_names,ldas=None,ret_topk=20):
    '''input a tuple of req,wsdl , return a seqs of recommender wsdls' names
    
    Arguments:
        w2v_model {[type]} -- [description]
        lstm_model {[type]} -- [description]
        input_seqs {[type]} -- [description]
        wsdl_names {[type]} -- [description]
    
    Keyword Arguments:
        ret_topk {int} -- [description] (default: {20})
    '''
    req_words=input_seqs[0] # each element is a list of words
    wsdl_words = input_seqs[1]
    vec_req = []
    for word in req_words:
        vec_req.append(vectorize_words(w2v_model,word))
    vec_wsdl = []
    for word in wsdl_words:
        vec_wsdl.append(vectorize_words(w2v_model,word))
    
    req_b,wsdl_b = input_tfs
    if ldas:
        req_lda,wsdl_lda = ldas
    else:
        req_lda=None
        wsdl_lda = None
    req_tensor = TMN_model(req_b,vec_req,req_lda).sum(dim=1)
    wsdl_tensor = TMN_model(wsdl_b,vec_wsdl,wsdl_lda)
    scores = cos(req_tensor.unsqueeze(1),wsdl_tensor.unsqueeze(0),dim=2)
    _,indices = scores.topk(ret_topk,dim=1) # along the dim 1
    wsdl_names = np.array(wsdl_names)
    return wsdl_names[indices.cpu()]


