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


def generateTrainAndTest():
    '''
     do 5 fold cross validation
     return train , test seqs
    '''
    # random the seqs for each invoke
    random.shuffle(seqs)
    total_len = len(seqs)
    fold_len = int(total_len/5)
    train1 = seqs[fold_len:]
    test1 = seqs[:fold_len]
    train2 = seqs[:fold_len]+seqs[2*fold_len:]
    test2 = seqs[fold_len:2*fold_len]
    train3 = seqs[:2*fold_len]+seqs[3*fold_len:]
    test3 = seqs[2*fold_len:3*fold_len]
    train4 = seqs[:3*fold_len]+seqs[4*fold_len:]
    test3 = seqs[3*fold_len:4*fold_len]
    train4 = seqs[:3*fold_len]+seqs[4*fold_len:]
    test4 = seqs[3*fold_len:4*fold_len]
    train5 = seqs[:4*fold_len]
    test5 = seqs[4*fold_len:]
    return train1, train2, train3, train4, train5, test1, test2, test3, test4, test5


class SimDataSet(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = torch.tensor(self.seqs[index][:-1])
        label = torch.tensor(self.seqs[index][-1])
        return seq, label
