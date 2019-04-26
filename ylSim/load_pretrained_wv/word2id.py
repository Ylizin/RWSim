import os 
import utils
import numpy as np
from .save_w2v import save_w2v_w2i

def load_w2vi(word2index=None,save_path = utils.extract_w2v_path):
    '''this function load the save_path+'w2i.txt' if word2index is not given or is an empty dict
        else if the word2index provided is not the same as cached, it will generate again
    
    Arguments:
        word2index {dict} -- word2index
    
    Keyword Arguments:
        save_path {[type]} -- [description] (default: {utils.extract_w2v_path})
    
    Returns:
        [type] -- [description]
    '''
    pret = None
    w2i = {}    

    if os.path.exists(save_path+'w2i.txt'):
        with open(save_path+'w2i.txt','r') as f:
            for no,line in enumerate(f):
                line = line.strip().split(',')
                if line:
                    word = line[1]
                    index = line[0]
                    w2i[word] = int(index)
    
    if not word2index: #if w2index not given 
        pret = __load_w2v()
    elif w2i != word2index:  # if the corpus updated ,the we need to regenerate the npy,here should be completed
        pret = save_w2v_w2i(word2index)
    else:
        pret = __load_w2v()
    return pret,w2i

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

if __name__ == '__main__':
    w2i = {}
    if os.path.exists(utils.extract_w2v_path+'w2.txt'):
        with open(utils.extract_w2v_path+'w2.txt','r') as f:
            for no,line in enumerate(f):
                line = line.strip().split(',')
                if line:
                    word = line[1]
                    index = line[0]
                    w2i[word] = int(index)  
    load_w2vi(w2i)  