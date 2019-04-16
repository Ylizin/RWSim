import os 

w2i = {}

def __load_w2i(save_path):
    with open(save_path+'w2i.txt','r') as f:
        for no,line in enumerate(f):
            line = line.strip().split()[0]
            w2i[line] = no

def word2index(word):
    if not w2i:
        __load_w2i('./test')
    index = -1
    try:
        index = w2i[word]
    except KeyError:
        pass
    finally:
        return index

