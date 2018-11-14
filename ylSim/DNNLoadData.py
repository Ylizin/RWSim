# -*- coding : utf-8 -*-
import os
import random

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

import loadRelevance
import utils

loadRelevance.loadRelevance()
relevanceDict = loadRelevance.relevanceDict

seqs = []


def loadFeatures(featurePath):
    for file in os.listdir(featurePath):
        fullpath = os.path.join(featurePath, file)
        if os.path.isdir(fullpath):
            continue

        fileDict = relevanceDict[file]
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
                seqs.append(feature)
    print('features reading complete')


def generateTrainAndTest(cvNum):
    '''
     do cvNum fold cross validation
     return train , test seqs
    '''
    # random the seqs for each invoke
    random.shuffle(seqs)
    total_len = len(seqs)
    fold_len = int(total_len/cvNum)
    train_testLists = []
    for i in range(1, cvNum+1):
        train = seqs[:(i-1)*fold_len] + seqs[i*fold_len:]
        test = seqs[(i-1)*fold_len:i*fold_len]
        train_testLists.append((train,test))

    return train_testLists


class SimDataSet(Dataset):
    def __init__(self, seqs, level = 3):
        self.seqs = seqs
        self.level = level

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = torch.tensor(self.seqs[index][:-1])
        label = torch.tensor(1 if self.seqs[index][-1]>self.level-0.1 else 0)
        return seq, label
