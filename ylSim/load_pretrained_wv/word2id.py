import os 
import utils
import numpy as np
from .save_w2v import save_w2v_w2i

def load_w2vi(word2index,save_path = utils.extract_w2v_path):
    w2i = {}
    if os.path.exists(save_path+'.txt'):
        with open(save_path+'.txt','r') as f:
            for no,line in enumerate(f):
                line = line.strip().split()
                if line:
                    word = line[0]
                    index = line[1]
                    w2i[word] = int(index)
    pret = None
    if w2i != word2index:  # if the corpus updated ,the we need to regenerate the npy,here should be completed
        pret = save_w2v_w2i(word2index)
    else:
        pret = __load_w2v()
    return w2i,pret

def __load_w2v(save_path = utils.extract_w2v_path):
    w2v = np.load(save_path+'.npy')
    return w2v

# def word2index(word):
#     if not w2i:
#         load_w2i(utils.extract_w2v_path+'.txt')
#     index = -1
#     try:
#         index = w2i[word]
#     except KeyError:
#         pass
#     finally:
#         return index

