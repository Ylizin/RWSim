import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


import loadRelevance
import utils
from LSTM.LoadData import (generateTrainAndTest, getSeqsFromKeys,
                           relevanceDict, reqFeatures, wsdlFeatures)

reqFeaturePath = utils.RQPath+r'/raw_vec'
wsdlFeaturePath = utils.WSDLPath+r'/raw_vec'

def concate_narr(n_arr,max_length):
    arr_length = n_arr.shape[0]
    if arr_length > max_length:
        n_arr = n_arr[:max_length]
    elif arr_length<max_length:
        n_arr = np.concatenate([n_arr,np.zeros((max_length-arr_length,utils.dw))],axis = 0)
    return n_arr

def loadFeatures(max_length,relevancePath= utils.RelevancePath, wsdlPath =utils.WSDLPath):
    '''for every req in rel_path, we load it , in CNN we padding them to the max_length*dw 
    
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
        n_arr = np.load(fullpath)
        n_arr = concate_narr(n_arr,max_length)
        reqFeatures[file] = n_arr
    
    for file in os.listdir(wsdlPath):
        fullpath = os.path.join(wsdlPath,file)
        if os.path.isdir(fullpath):
            continue
        fullpath = os.path.join(wsdlFeaturePath, file+'.npy')
        n_arr = np.load(fullpath)
        n_arr = concate_narr(n_arr,max_length)
        wsdlFeatures[file] = n_arr
    print('features reading complete')

class CNNDataSet(Dataset):
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

class CNNDataLoader(object):
    def __init__(self, CNNDataSet,batch_size = 128):
        self.data = CNNDataSet
        self.len = len(CNNDataSet) 
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