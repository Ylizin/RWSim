import os
import random

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

import utils
import loadRelevance
from load_pretrained_wv.word2id import load_w2vi

relevanceDict = {}
rq_words = {}
wsdl_words = {}

rq_word_path = utils.RQPath
wsdl_word_path = utils.WSDLPath 
w2vi_path = utils.extract_w2v_path 


def load_words(relevancePath= utils.RelevancePath, wsdlPath =utils.WSDLPath):
    loadRelevance.loadRelevance()
   
    global relevanceDict
    relevanceDict.update(loadRelevance.relevanceDict)

    rq_files = utils.iterate_path(relevancePath)
    wsdl_files = utils.iterate_path(wsdlPath)

    for file in rq_files:
        full_path = os.path.join(rq_word_path,file)
        words = []
        with open(full_path ,'r') as f:
            for line in f:
                line = line.strip().split()
                words.extend(line)
        rq_words[file] = words
    
    for file in wsdl_files:
        full_path = os.path.join(wsdl_word_path,file)
        words = []
        with open(full_path,'r') as f:
            for line in f:
                line = line.strip().split()
                words.extend(line)
        wsdl_words[file] = words

    print('words load complete.')

def generateTrainAndTest(cvNum):
    seqs_keys = list(rq_words.keys())
    random.shuffle(seqs_keys)
    total_len = len(seqs_keys)
    fold_len = int(total_len/cvNum)
    train_testLists = []
    for i in range(1, cvNum+1):
        train_keys = seqs_keys[:(i-1)*fold_len] + seqs_keys[i*fold_len:]
        test_keys = seqs_keys[(i-1)*fold_len:i*fold_len]
        train_testLists.append((train_keys,test_keys))
    return train_testLists


def get_words_from_keys(keys):
    if not rq_words:
        load_words()

    if isinstance(keys,str) : #if the param is a single str 
        keys = [keys]
    random.shuffle(keys)
    return_seqs = []

    for key in keys:
        for wsdl in wsdl_words:
            rq_word = rq_words[key]
            wsdl_word = wsdl_words[wsdl]
            rel = 0
            rel = utils.get_relLevel(relevanceDict,key,wsdl)
            return_seqs.append((rq_word,wsdl_word,rel))
    
    return return_seqs

#the word out of vocab is a serious defect for such a tiny dataset
#here the startegy should be modified for those oovs 
class LSTMDataSet(Dataset):
    def __init__(self,seqs,eval = True):
        if eval:
            self.seqs = seqs
        else:
            self.seqs = seqs + self.__swap_req_wsdl(seqs)
        self.__make_dict()
        self.pret = load_w2vi(self.word2index)
        self.seqs = self.__seqs2index(self.seqs)

    def __swap_req_wsdl(self,seqs):
        new_seqs = []
        for reqF,wsdlF,rel in seqs:
            new_seqs.append((wsdlF,reqF,rel))
        return new_seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        
        req,wsdl,label = zip(*self.seqs[index])
        #we do not convert them into the tensor here
        # req = torch.from_numpy(reqF)
        # wsdl = torch.from_numpy(wsdlF)
        # label = torch.LongTensor(rel,)
        return req,wsdl,label
    
    def __make_dict(self):
        dic = set()
        for rq,wsdl,rel in self.seqs:
            dic.update(rq)
            dic.update(wsdl)
        self.word2index = {}
        for i,word in enumerate(dic):
            self.word2index[word] = i

    def __seqs2index(self,seqs):
        index_seqs = []
        for rq_word,wsdl_word,rel in seqs:
            rq_index = []
            wsdl_index = []
            for word in rq_word:
                rq_index.append(self.word2index[word])
            for word in wsdl_word:
                wsdl_index.append(self.word2index[word])
            index_seqs.append((rq_index,wsdl_index,rel))
        return index_seqs


class LSTMDataLoader(object):
    def __init__(self, LSTMDataSet,batch_size = 128):
        self.data = LSTMDataSet
        self.len = len(LSTMDataSet) 
        self.batch_size = batch_size
        self._idx = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        idx = self._idx
        batch_size = self.batch_size
        #when idx == length means it runs out of the dataset
        if idx >= self.len:
            self._idx = 0
            raise StopIteration
        upper_bound = self.len if self.len <= idx + batch_size else idx + batch_size
        self._idx += self.batch_size
        return self.data[idx:upper_bound]
        








