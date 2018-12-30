# -*- coding : utf-8 -*-
import sys

sys.path.append('..')

import os
import random

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

import loadRelevance
import utils

relevanceDict = {}
seqs = {}

def loadFeatures(featurePath):
    loadRelevance.loadRelevance()
    global relevanceDict
    relevanceDict.update(loadRelevance.relevanceDict)

    for file in os.listdir(featurePath):
        fullpath = os.path.join(featurePath, file)
        if os.path.isdir(fullpath):
            continue
        fileDict = relevanceDict[file]
        seq = []
        with open(fullpath, 'r') as f:
            for line in f:
                data = line.strip().split()
                name = data[0]
                feature = data[1:]
                feature = list(map(float, feature))  # convert to float
                thisHighReq = fileDict['highRelevance']
                thisMidReq = fileDict['midRelevance']
                thisLowReq = fileDict['lowRelevance']
                if name in thisHighReq:
                    feature.append(3)
                elif name in thisMidReq:
                    feature.append(2)
                elif name in thisLowReq:
                    feature.append(1)
                else:
                    feature.append(0)
                seq.append(feature)
        seqs[file] = seq
    print('features reading complete')

def generateTrainAndTest(cvNum):
    '''
     do cvNum fold cross validation
     return train , test seqs
    '''
    seqs_keys = list(seqs.keys())

    # random the seqs for each invoke
    random.shuffle(seqs_keys)
    total_len = len(seqs_keys)
    fold_len = int(total_len/cvNum)
    train_testLists = []
    for i in range(1, cvNum+1):
        train_keys = seqs_keys[:(i-1)*fold_len] + seqs_keys[i*fold_len:]
        test_keys = seqs_keys[(i-1)*fold_len:i*fold_len]
        train_testLists.append((train_keys,test_keys))
    return train_testLists

def getSeqsFromKeys(keys):
    '''
       careful that for evaluation metrics procedure, requests should be test separately

    '''
    if len(seqs) == 0:
        loadFeatures(utils.featurePath)

    if isinstance(keys,str) : #if the param is a single str 
        keys = [keys]
    key_seqs = []
    for key in keys:
        key_seqs += seqs[key]
    random.shuffle(key_seqs)

    return key_seqs

def getLen(reqName,level):
    if len(seqs) == 0:
        loadFeatures(utils.featurePath)
    fileDict = relevanceDict[reqName]
    thisHighReq = fileDict['highRelevance']
    thisMidReq = fileDict['midRelevance']
    thisLowReq = fileDict['lowRelevance']
    len_high = len(thisHighReq)
    len_mid = len(thisMidReq)
    len_low = len(thisLowReq)
    if level == 3:
        return len_high
    elif level == 2:
        return len_high+len_mid
    else:
        return len_high+len_mid+len_low

class SimDataSet(Dataset):
    def __init__(self, seqs, level = 3):
        self.seqs = seqs
        self.level = level

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = torch.tensor(self.seqs[index][:-1])
        #for binary classification
        # label = torch.tensor(1 if self.seqs[index][-1]>self.level-0.1 else 0)

        label = torch.FloatTensor(self.seqs[index][-1:])
        return seq, label
