
import os
import random

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

import loadRelevance
import utils
from LSTM.LoadData import (generateTrainAndTest, loadFeatures, relevanceDict,
                           reqFeatures, wsdlFeatures)
from LSTM.trainLSTM import calculateLevelsPN

# relevanceDict = {}
# reqFeatures = {}
# wsdlFeatures = {}
reqBows = {}
wsdlBows = {}

reqFeaturePath = utils.RQPath+r'\raw_vec'
wsdlFeaturePath = utils.WSDLPath+r'\raw_vec'


def loadBow(rqPath=utils.RQ_TF_path, wsdlPath=utils.WSDL_TF_path):
    '''load the req and wsdl BoWs 

    Keyword Arguments:
        relevancePath {[type]} -- [description] (default: {utils.RQ_TF_path})
        wsdlPath {[type]} -- [description] (default: {utils.WSDL_TF_path})
    '''
    global reqBows
    global wsdlBows
    RQ_paths = utils.iterate_path(rqPath)
    for file in RQ_paths:
        bow = []
        with open(os.path.join(rqPath, file), 'r') as f:
            for line in f:
                line = line.strip().split()
                for i_f in line:
                    idx, freq = i_f.split(',')
                    idx = int(idx)
                    freq = int(freq)
                    bow.append((idx, freq))
        reqBows[file] = bow

    WSDL_paths = utils.iterate_path(wsdlPath)
    for file in WSDL_paths:
        bow = []
        with open(os.path.join(wsdlPath, file), 'r') as f:
            for line in f:
                line = line.strip().split()
                for i_f in line:
                    idx, freq = i_f.split(',')
                    idx = int(idx)
                    freq = int(freq)
                    bow.append((idx, freq))
        wsdlBows[file] = bow


def getSeqsFromKeys(keys):
    '''
       careful that for evaluation metrics procedure, requests should be test separately

    '''
    if not reqFeatures:
        loadFeatures()
    if not reqBows:
        loadBow()

    if isinstance(keys, str):  # if the param is a single str
        keys = [keys]
    random.shuffle(keys)
    return_seqs = []

    for req in keys:
        for wsdl in wsdlFeatures.keys():
            reqBow = reqBows[req]
            reqF = reqFeatures[req]
            wsdlBow = wsdlBows[wsdl]
            wsdlF = wsdlFeatures[wsdl]
            rel = 0
            rel = utils.get_relLevel(relevanceDict, req, wsdl)
            return_seqs.append((reqBow, reqF, wsdlBow, wsdlF, rel))

    return return_seqs


def getAllBows():
    if not reqBows:
        loadBow()
    return_seqs = []
    for key in reqBows:
        return_seqs.append(reqBows[key])
    for key in wsdlBows:
        return_seqs.append(wsdlBows[key])
    return_tuples = []
    for seq in return_seqs:
        return_tuples.append((seq,)*5)
    return return_tuples


class NTMDataSet(Dataset):
    def __init__(self, seqs, eval=True):
        if eval:
            self.seqs = seqs
        else:
            self.seqs = seqs + self.__swap_req_wsdl(seqs)

    def __swap_req_wsdl(self, seqs):
        new_seqs = []
        for reqF, wsdlF, rel in seqs:
            new_seqs.append((wsdlF, reqF, rel))
        return new_seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):

        req_b, req, wsdl_b, wsdl, label = zip(*self.seqs[index])
        # we do not convert them into the tensor here
        # req = torch.from_numpy(reqF)
        # wsdl = torch.from_numpy(wsdlF)
        # label = torch.LongTensor(rel,)
        return req_b, req, wsdl_b, wsdl, label


class NTMDataLoader(object):
    def __init__(self, NTMDataSet, batch_size=128):
        self.data = NTMDataSet
        self.len = len(NTMDataSet)
        self.batch_size = batch_size
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        idx = self._idx
        batch_size = self.batch_size
        # when idx == length means it runs out of the dataset
        if idx >= self.len:
            self._idx = 0
            raise StopIteration
        upper_bound = self.len if self.len <= idx + batch_size else idx + batch_size
        self._idx += self.batch_size
        return self.data[idx:upper_bound]


if __name__ == '__main__':
    loadFeatures()
    lists = generateTrainAndTest(5)
    test_list = lists[2]
    test_set = NTMDataSet(getSeqsFromKeys(test_list[0]))
    test_loader = NTMDataLoader(test_set)

    for req_d, req, wsdl_d, wsdl, rel in test_loader:
        print(len(req_d))
        print(len(req_d[0]))
