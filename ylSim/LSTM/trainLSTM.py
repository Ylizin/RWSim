import sys

sys.path.append('..')

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
import LSTM 
import LoadData
import utils
from LSTM import RWLSTM

_CUDA = torch.cuda.is_available()

def simplePrecisionNDCG(reqName, pred_r, topK=5, level=3, doDCG=False):
    '''
        pred_r is sorted and cut out topK 
        if nr < k then k = nr
    '''
    tp = 0
    fp = 1
    DCG = 0.0
    IDCG = 1
    len_p = DNNLoadData.getLen(reqName,level)
    topK = topK if topK<len_p else len_p
    for i, t in enumerate(pred_r):
        if i > topK :
            break
        pred, r = t
        if doDCG:
            DCG += calculatePrecision.calculateDCG(r, i+1, K1=i+1)
        if r < level:  # eg.  here we have a r=2 but ecpected level 3
            fp += 1
        else:
            tp += 1
    if doDCG:
        IDCG = calculatePrecision.calculateIDCG(reqName, topK)
    if fp > 1:
        fp -= 1
    return tp/(fp+tp), DCG/IDCG

def trainOneModel(args, model, trainSeqs, testSeqs, testSeqs_keys, index, syncCount, syncPrecision1, syncPrecision2, syncPrecision3, syncNDCG, lock, doPrint=False):
    if index % 5 == 0:
        doPrint = True
    level = args.level
    topK = 5

    trainDataset = DNNLoadData.SimDataSet(trainSeqs, level)
    testDataset = DNNLoadData.SimDataSet(testSeqs, level)

    trainDataLoader = DataLoader(
        trainDataset, args.batchSize, num_workers=args.numWorkers)
    testDataloader = DataLoader(
        testDataset, args.batchSize, num_workers=args.numWorkers)

    # 1, 5, 5, 5 is a nice weight for rrelu
    lossWeight = torch.tensor([10.0])
    if _CUDA:
        torch.cuda.set_device(0)
        model = model.cuda()
        # default GPU is 0
        lossWeight = lossWeight.cuda()

    # add weight to emphasize the high relevance case
    lossFunc = customizedLoss
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    bestPrecision = 0.0
    for i in range(args.nepoch):
        totalLoss = 0.0
        scheduler.step()

        model.train()
        # attention! here if u are on Windows, the --numWorker should not be too large otherwise it will overconsume the memory
        for seq, r in trainDataLoader:

            if _CUDA:
                seq = seq.cuda()
                r = r.cuda()
            r.view(-1)
            pred = model(seq)

            l = lossFunc(pred, r) * lossWeight
            totalLoss += l.item()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        if doPrint:
            print('epoch:{},Training loss :{:.4}'.format(
                i, totalLoss))
        if i % args.testEvery == (args.testEvery - 1):
            precision1 = 0.0
            precision2 = 0.0
            precision3 = 0.0
            NDCG = 0.0
            model.eval()

            for key in testSeqs_keys:  # do evaluation for every key respectively
                predicts = []
                evalSeqs = DNNLoadData.getSeqsFromKeys(key)
                evalDataSet = DNNLoadData.SimDataSet(evalSeqs)
                evalDataloader = DataLoader(
                    evalDataSet, args.batchSize, num_workers=args.numWorkers)
                for seq, r in evalDataloader:
                    if _CUDA:
                        seq = seq.cuda()
                        r = r.cuda()
                    r = r.view(-1)
                    pred = model(seq)
                    pred = pred.view(-1)
                    predicts += list(zip(pred, r))  # list of (predict,r)
                sortedResult = sorted(
                    predicts, key=lambda k: k[0], reverse=True)
                p1, ndcg = simplePrecisionNDCG(
                    key, sortedResult[:topK], topK, 1, doDCG=True)
                p2, _ = simplePrecisionNDCG(
                    key, sortedResult[:topK], topK, 2, doDCG=True)
                p3, _ = simplePrecisionNDCG(
                    key, sortedResult[:topK], topK, 3, doDCG=True)
                precision1 += p1
                precision2 += p2
                precision3 += p3
                NDCG += ndcg
            precision1 = precision1/len(testSeqs_keys)
            precision2 = precision2/len(testSeqs_keys)
            precision3 = precision3/len(testSeqs_keys)
            NDCG = NDCG/len(testSeqs_keys)
            NDCG = NDCG.item()
            if precision1 > bestPrecision:
                torch.save(model.state_dict(), args.modelFile +
                           str(level)+str(index))
                bestPrecision = precision1
            if doPrint:
                print('epoch:{},Precision1:{:.4},Precision2:{:.4},Precision3:{:.4},NDCG:{:.4}'.format(
                    i, precision1, precision2, precision3, NDCG))

    if doPrint:
        print('bestPrecision:{}'.format(bestPrecision))
    model.load_state_dict(torch.load(args.modelFile+str(level)+str(index)))

    precision1 = 0.0
    precision2 = 0.0
    precision3 = 0.0
    NDCG = 0.0
    model.eval()

    for key in testSeqs_keys:  # do evaluation for every key respectively
        predicts = []
        evalSeqs = DNNLoadData.getSeqsFromKeys(key)
        evalDataSet = DNNLoadData.SimDataSet(evalSeqs)
        evalDataloader = DataLoader(
            evalDataSet, args.batchSize, num_workers=args.numWorkers)
        for seq, r in evalDataloader:
            if _CUDA:
                seq = seq.cuda()
                r = r.cuda()
            r = r.view(-1)
            pred = model(seq)
            pred = pred.view(-1)
            predicts += list(zip(pred, r))  # list of (predict,r)
        sortedResult = sorted(predicts, key=lambda k: k[0], reverse=True)
        p1, ndcg = simplePrecisionNDCG(
            key, sortedResult[:topK], topK, 1, doDCG=True)
        p2, _ = simplePrecisionNDCG(
            key, sortedResult[:topK], topK, 2, doDCG=True)
        p3, _ = simplePrecisionNDCG(
            key, sortedResult[:topK], topK, 3, doDCG=True)
        precision1 += p1
        precision2 += p2
        precision3 += p3
        NDCG += ndcg

    precision1 = precision1/len(testSeqs_keys)
    precision2 = precision2/len(testSeqs_keys)
    precision3 = precision3/len(testSeqs_keys)

    NDCG = NDCG/len(testSeqs_keys)
    NDCG = NDCG.item()
    with lock:
        syncCount.value += 1
        syncPrecision1.value += precision1
        syncPrecision2.value += precision2
        syncPrecision3.value += precision3
        syncNDCG.value += NDCG
    # return precision

