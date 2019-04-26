import warnings
import utils
warnings.filterwarnings(action="ignore", category=UserWarning, module="gensim")
import gensim
from gensim.models import KeyedVectors

import numpy as np
import os

def load_w2v(path):
    model = KeyedVectors.load_word2vec_format(path, binary=True)
    return model

# def __extract_vocab(model):
#     model_vocab = model.vocab
#     return model_vocab

# def __extrac_word2index(vocab,word_dict):
#     w2i = []
#     for word in word_dict:
#         w2i.append((word,vocab[word].index))
#     return w2i

def w2v(model,word):
    try:
        return model[word] #return shape (1,dw)
    except KeyError:
        return np.zeros(utils.dw)

def save_w2v_w2i(word_dict,pret_path = utils.google_pretrained_path,save_path = utils.extract_w2v_path):
    '''here we generate an numpy array for the input word_dict
    
    Arguments:
        word_dict {dict} -- a dict containing the {word(str):index(int)}
    
    Keyword Arguments:
        pret_path {str} -- [description] (default: {utils.google_pretrained_path})
        save_path {[type]} -- [description] (default: {utils.extract_w2v_path})
    '''

    global __saved_path
    model = load_w2v(pret_path)
    vec_list = []
    for word in word_dict:
        w_v = w2v(model,word)
        if w_v is None:
            continue
        else:
            vec_list.append(w_v)
    
    pret_np = np.vstack(vec_list) #this should be len(vocab)*dw
    np.save(save_path,pret_np)
    
    with open(save_path+'w2i.txt', 'w') as f:
        for word in word_dict:
            f.write(str(word_dict[word])+','+word+'\n')
    return pret_np


        