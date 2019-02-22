import argparse
import os

import traceback
from multiprocessing import Manager, Pool

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader

import calculatePrecision
import LSTM.LoadData as LoadData
import LSTM.transformer_totalModel as totalmodel
import utils
from calculatePrecision import getLen
from LSTM.transformer_totalModel import RWTSMModel

from LSTM.trainLSTM import calculateLevelsPN,customizedLoss,customizedLoss2,trainOneModel

def main():
    parser = argparse.ArgumentParser("LSTM")
    parser.add_argument("--outDim", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=150)
    
    # parser.add_argument('--hiddenDim2', type=int, default=60)
    # parser.add_argument('--hiddenDim3', type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--bidirectional", type=bool, default=True)
    
    # parser.add_argument('--numWorkers', type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--foldNum", type=int, default=5)
    parser.add_argument("--level", type=int, default=3)

    parser.add_argument("--nepoch", type=int, default=300)
    parser.add_argument("--testEvery", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--modelFile", default="./models/LSTM")

    args = parser.parse_args()

    LoadData.loadFeatures()

    train_test_Seqs = LoadData.generateTrainAndTest(args.foldNum)
    # level has 1,2,3 each level we train foldNum models
    TSMModels = [RWTSMModel(args) for i in range(args.foldNum)]
    level = args.level

    manager = Manager()
    p = Pool(int(1))  # transformer is too large
    lock = manager.Lock()
    precision1 = manager.Value("d", 0.0)
    precision2 = manager.Value("d", 0.0)
    precision3 = manager.Value("d", 0.0)
    NDCG = manager.Value("d", 0.0)
    count = manager.Value("i", 0)
    # testSetPrecision = []
    for index, model in enumerate(TSMModels):
        # get the index fold train and test seqs
        ttSeq = train_test_Seqs[index]
        trainSeqs_keys, testSeqs_keys = ttSeq
        trainSeqs = LoadData.getSeqsFromKeys(trainSeqs_keys)
        testSeqs = LoadData.getSeqsFromKeys(testSeqs_keys)
        p.apply_async(
            trainOneModel,
            args=(
                args,
                model,
                trainSeqs,
                testSeqs,
                trainSeqs_keys,
                testSeqs_keys,
                index,
                count,
                precision1,
                precision2,
                precision3,
                NDCG,
                lock,
            ),
            error_callback=utils.errorCallBack,
        )
        # precision = trainOneModel(args,model,trainSeqs,testSeqs,level,index)
        # testSetPrecision.append(precision)
    p.close()
    p.join()
    count = count.value
    precision1 = precision1.value / count
    precision2 = precision2.value / count
    precision3 = precision3.value / count
    NDCG = NDCG.value / count
    print(
        str(args.foldNum) + "foldCV precision1:" + str(precision1)
    )  # +str(np.mean(testSetPrecision)))
    print("precision2 :{}".format(precision2))
    print("precision3 :{}".format(precision3))
    print("NDCG : {}".format(NDCG))

