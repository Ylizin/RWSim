import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


import loadRelevance
import utils
from LSTM.LoadData import (generateTrainAndTest,
                           relevanceDict, reqFeatures, wsdlFeatures)
from LSTM.trainLSTM import calculateLevelsPN,customizedLoss2


reqFeaturePath = utils.RQPath+r'/raw_vec'
wsdlFeaturePath = utils.WSDLPath+r'/raw_vec'

def concate_narr(n_arr,max_length):
    arr_length = n_arr.shape[0]
    if arr_length > max_length:
        n_arr = n_arr[:max_length]
    elif arr_length<max_length:
        n_arr = np.concatenate([n_arr,np.zeros((max_length-arr_length,utils.dw))],axis = 0)
    return n_arr

def concate_idx(li,max_length):
    li_length = len(li)
    if li_length>max_length:
        li = li[:max_length]
    elif li_length<max_length:
        padding_idxs = [-1 for _ in range(max_length-li_length)]
        li.extend(padding_idxs)
    return li

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

def load_idx(max_length,rq_idx_path = utils.RQ_IDX_Path,wsdl_idx_path = utils.WSDL_IDX_Path):
    '''this function will read the idx of each doc and return the wsdl and req docs' indexs
    and the padding idx is -1
    
    Keyword Arguments:
        rq_idx_path {[type]} -- [description] (default: {utils.RQ_IDX_Path})
        wsdl_idx_path {[type]} -- [description] (default: {utils.WSDL_IDX_Path})
    
    Returns:
        dict -- {file:list of int}
    '''
    rq_idx = {}
    wsdl_idx = {}
    
    rq_idx_paths = utils.iterate_path(rq_idx_path)
    wsdl_idx_paths = utils.iterate_path(wsdl_idx_path)

    for file in rq_idx_paths:
        full_path = os.path.join(rq_idx_path,file)
        with open(full_path,'r') as f:
            for line in f:
                data = line.strip().split()
                data = list(map(int,data))
                data = concate_idx(data,max_length)
                rq_idx[file] = data

    for file in wsdl_idx_paths:
        full_path = os.path.join(wsdl_idx_path,file)
        with open(full_path,'r') as f:
            for line in f:
                data = line.strip().split()
                data = list(map(int,data))
                data = concate_idx(data,max_length) 
                wsdl_idx[file]= data

    return rq_idx,wsdl_idx     
    
def getSeqsFromKeys(keys,max_length):
    '''
       careful that for evaluation metrics procedure, requests should be test separately

    '''
    if len(reqFeatures) == 0:
        loadFeatures(max_length)

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
        #torch tensor can convert a list of numpy array into tensor
        req,wsdl,label = zip(*self.seqs[index])
        req = torch.tensor(req,dtype = torch.float)
        wsdl = torch.tensor(wsdl,dtype = torch.float)
        label = torch.tensor(label)
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