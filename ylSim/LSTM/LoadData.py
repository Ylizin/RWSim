import sys

sys.path.append('..')

import os
import random

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

import utils
import loadRelevance

relevanceDict = {}
reqFeatures = {}
wsdlFeatures = {}

reqFeaturePath = utils.RQPath+r'/raw_vec'
wsdlFeaturePath = utils.WSDLPath+r'/raw_vec'

def loadFeatures(relevancePath= utils.RelevancePath, wsdlPath =utils.WSDLPath):
    '''for every req in rel_path, we load it 
    
    Keyword Arguments:
        relevancePath {[type]} -- [description] (default: {utils.RelevancePath})
        wsdlPath {[type]} -- [description] (default: {utils.WSDLPath})
    '''
    loadRelevance.loadRelevance()
   
    global relevanceDict
    relevanceDict.update(loadRelevance.relevanceDict)

    for file in os.listdir(relevancePath):
        fullpath = os.path.join(relevancePath,file)
        if os.path.isdir(fullpath):
            continue    
        fullpath = os.path.join(reqFeaturePath, file+'.npy')
        reqFeatures[file] = np.load(fullpath)
    
    for file in os.listdir(wsdlPath):
        fullpath = os.path.join(wsdlPath,file)
        if os.path.isdir(fullpath):
            continue
        fullpath = os.path.join(wsdlFeaturePath, file+'.npy')
        wsdlFeatures[file] = np.load(fullpath)
    print('features reading complete')

def generateTrainAndTest(cvNum,use_saved_seqs = False):
    '''
     do cvNum fold cross validation
     return train , test seqs
    '''
    seqs_keys=[[] for i in range(3)]
    seqs_lens = [] # according to the image, we separate three layers, 0-25,25-45,45-
    for rq in reqFeatures:
        len_ = reqFeatures[rq].shape[0]
        if len_<25:
            seqs_keys[0].append(rq)
        elif len_<45:
            seqs_keys[1].append(rq)
        else:
            seqs_keys[2].append(rq)
        # seqs_keys.append(rq)
        # seqs_lens.append(len_)
    # plt.hist(seqs_lens)
    # plt.savefig('img.png',format='png')
    if use_saved_seqs:
        idx_keys = []
        with open('./models/test_seqs','r') as f:
            for line in f:
                data = line.strip().split(':')
                index = int(data[0][1]) #get the index num
                test_keys = eval(data[1])
                idx_keys.append(test_keys)
        train_testLists = []
        for idx,test_keys in enumerate(idx_keys):
            train_keys = filter(lambda x: x not in test_keys , seqs_keys)
            train_keys = list(train_keys)
            train_testLists.append((train_keys,test_keys))
        return train_testLists


    # random the seqs for each invoke
    train_testLists = [[[],[]] for i in range(cvNum)]
    for s_k in seqs_keys:
        random.shuffle(s_k)
        total_len = len(s_k)
        
        total_len = len(s_k)        
        fold_len = int(total_len/cvNum)
        for i in range(1, cvNum+1):
            train_keys = s_k[:(i-1)*fold_len] + s_k[i*fold_len:]
            test_keys = s_k[(i-1)*fold_len:i*fold_len]
            train_testLists[i-1][0]+=train_keys
            train_testLists[i-1][1] += test_keys
    return train_testLists

def getSeqsFromKeys(keys):
    '''
       careful that for evaluation metrics procedure, requests should be test separately

    '''
    if len(reqFeatures) == 0:
        loadFeatures()

    if isinstance(keys,str) : #if the param is a single str 
        keys = [keys]
    random.shuffle(keys)
    return_seqs = []
    
    for req in keys:
        for wsdl in wsdlFeatures:
            reqF = reqFeatures[req]
            wsdlF = wsdlFeatures[wsdl]
            rel = 0
            rel = utils.get_relLevel(relevanceDict,req,wsdl)
            return_seqs.append((reqF,wsdlF,rel))

    return return_seqs


class LSTMDataSet(Dataset):
    def __init__(self,seqs,eval = True):
        if eval:
            self.seqs = seqs
        else:
            self.seqs = seqs + self.__swap_req_wsdl(seqs)

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
        
