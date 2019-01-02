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
import LSTM.totalmodel as totalmodel
import utils
from calculatePrecision import getLen
from LSTM.totalmodel import RWLSTMModel


_CUDA = torch.cuda.is_available()


def simplePrecisionNDCG(reqName, pred_r, topK=5, level=3, doDCG=False):
    """
        pred_r is sorted and cut out topK 
        if nr < k then k = nr
    """
    tp = 0
    fp = 1
    DCG = 0.0
    IDCG = 1
    len_p = getLen(reqName, level)
    topK = topK if topK < len_p else len_p
    for i, t in enumerate(pred_r):
        if i > topK:
            break
        pred, r = t
        if doDCG:
            DCG += calculatePrecision.calculateDCG(r, i + 1, K1=i + 1)
        if r < level:  # eg.  here we have a r=2 but ecpected level 3
            fp += 1
        else:
            tp += 1
    if doDCG:
        IDCG = calculatePrecision.calculateIDCG(reqName, topK)
    if fp > 1:
        fp -= 1
    return tp / (fp + tp), DCG / IDCG


def trainOneModel(
    args,
    model,
    trainSeqs,
    testSeqs,
    testSeqs_keys,
    index,
    syncCount,
    syncPrecision1,
    syncPrecision2,
    syncPrecision3,
    syncNDCG,
    lock,
    doPrint=False,
):
    if index % 5 == 0:
        doPrint = True
    level = args.level
    topK = 5

    trainDataset = LoadData.LSTMDataSet(trainSeqs)
    testDataset = LoadData.LSTMDataSet(testSeqs)

    trainDataLoader = LoadData.LSTMDataLoader(trainDataset)
    testDataloader = LoadData.LSTMDataLoader(testDataset)

    # 1, 5, 5, 5 is a nice weight for rrelu
    # lossWeight = torch.tensor([10.0])
    if _CUDA:
        torch.cuda.set_device(0)
        model = model.cuda()
        # default GPU is 0
        # lossWeight = lossWeight.cuda()

    # add weight to emphasize the high relevance case
    lossFunc = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    bestPrecision = 0.0
    for i in range(args.nepoch):
        totalLoss = 0.0
        scheduler.step()

        model.train()
        # attention! here if u are on Windows, the --numWorker should not be too large otherwise it will overconsume the memory
        for seq1, seq2, r in trainDataLoader:

            r = torch.tensor(r)
            if _CUDA:
                r = r.cuda()
            r.view(-1)
            pred = model(seq1, seq2)
            
            l = lossFunc(pred, r)
            totalLoss += l.item()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        if doPrint:
            print("epoch:{},Training loss :{:.4}".format(i, totalLoss))
        if i % args.testEvery == (args.testEvery - 1):
            precision1 = 0.0
            precision2 = 0.0
            precision3 = 0.0
            NDCG = 0.0
            model.eval()

            for key in testSeqs_keys:  # do evaluation for every key respectively
                predicts = []
                evalSeqs = LoadData.getSeqsFromKeys(key)
                evalDataSet = LoadData.LSTMDataSet(evalSeqs)
                evalDataloader = LoadData.LSTMDataLoader(evalDataSet)
                for seq1, seq2, r in evalDataloader:
                    r = torch.tensor(r)
                    if _CUDA:
                        r = r.cuda()
                    r = r.view(-1)
                    pred = model(seq1, seq2)
                    pred = nn.functional.softmax(pred, dim=1)
                    prob, predIndex_long = torch.max(pred, dim=1)
                    predIndex = predIndex_long.type_as(prob)
                    pred = torch.add(predIndex, prob)
                    r = r.type_as(pred)
                    # sort by pred , calculate by r
                    predicts += list(zip(pred, r))  # list of (predict,r)
                sortedResult = sorted(predicts, key=lambda k: k[0], reverse=True)
                p1, ndcg = simplePrecisionNDCG(
                    key, sortedResult[:topK], topK, 1, doDCG=True
                )
                p2, _ = simplePrecisionNDCG(
                    key, sortedResult[:topK], topK, 2, doDCG=True
                )
                p3, _ = simplePrecisionNDCG(
                    key, sortedResult[:topK], topK, 3, doDCG=True
                )
                precision1 += p1
                precision2 += p2
                precision3 += p3
                NDCG += ndcg
            precision1 = precision1 / len(testSeqs_keys)
            precision2 = precision2 / len(testSeqs_keys)
            precision3 = precision3 / len(testSeqs_keys)
            NDCG = NDCG / len(testSeqs_keys)
            NDCG = NDCG.item()
            if precision1 > bestPrecision:
                utils.generateDirs(args.modelFile)
                torch.save(model.state_dict(), args.modelFile + str(level) + str(index))
                bestPrecision = precision1
            if doPrint:
                print(
                    "epoch:{},Precision1:{:.4},Precision2:{:.4},Precision3:{:.4},NDCG:{:.4}".format(
                        i, precision1, precision2, precision3, NDCG
                    )
                )

    if doPrint:
        print("bestPrecision:{}".format(bestPrecision))
    model.load_state_dict(torch.load(args.modelFile + str(level) + str(index)))

    precision1 = 0.0
    precision2 = 0.0
    precision3 = 0.0
    NDCG = 0.0
    model.eval()

    for key in testSeqs_keys:  # do evaluation for every key respectively
        predicts = []
        evalSeqs = LoadData.getSeqsFromKeys(key)
        evalDataSet = LoadData.LSTMDataSet(evalSeqs)
        evalDataloader = LoadData.LSTMDataLoader(evalDataSet)
        for seq1, seq2, r in evalDataloader:
            r = torch.tensor(r)
            if _CUDA:
                r = r.cuda()
            r = r.view(-1)
            pred = model(seq1, seq2)
            pred = nn.functional.softmax(pred, dim=1)
            prob, predIndex_long = torch.max(pred, dim=1)
            predIndex = predIndex_long.type_as(prob)
            pred = torch.add(predIndex, prob)
            r = r.type_as(pred)
            # sort by pred , calculate by r
            predicts += list(zip(pred, r))  # list of (predict,r)
        sortedResult = sorted(predicts, key=lambda k: k[0], reverse=True)
        p1, ndcg = simplePrecisionNDCG(key, sortedResult[:topK], topK, 1, doDCG=True)
        p2, _ = simplePrecisionNDCG(key, sortedResult[:topK], topK, 2, doDCG=True)
        p3, _ = simplePrecisionNDCG(key, sortedResult[:topK], topK, 3, doDCG=True)
        precision1 += p1
        precision2 += p2
        precision3 += p3
        NDCG += ndcg

    precision1 = precision1 / len(testSeqs_keys)
    precision2 = precision2 / len(testSeqs_keys)
    precision3 = precision3 / len(testSeqs_keys)

    NDCG = NDCG / len(testSeqs_keys)
    NDCG = NDCG.item()
    with lock:
        syncCount.value += 1
        syncPrecision1.value += precision1
        syncPrecision2.value += precision2
        syncPrecision3.value += precision3
        syncNDCG.value += NDCG
    # return precision


def main():

    parser = argparse.ArgumentParser("DNN")
    parser.add_argument("--outDim", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=60)
    # parser.add_argument('--hiddenDim2', type=int, default=60)
    # parser.add_argument('--hiddenDim3', type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # parser.add_argument('--numWorkers', type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--foldNum", type=int, default=5)
    parser.add_argument("--level", type=int, default=3)

    parser.add_argument("--nepoch", type=int, default=300)
    parser.add_argument("--testEvery", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--modelFile", default="./models/LSTM")

    args = parser.parse_args()

    LoadData.loadFeatures()

    train_test_Seqs = LoadData.generateTrainAndTest(args.foldNum)
    # level has 1,2,3 each level we train foldNum models
    LSTMModels = [RWLSTMModel(args) for i in range(args.foldNum)]
    level = args.level

    manager = Manager()
    p = Pool(int(os.cpu_count() / 2))
    lock = manager.Lock()
    precision1 = manager.Value("d", 0.0)
    precision2 = manager.Value("d", 0.0)
    precision3 = manager.Value("d", 0.0)
    NDCG = manager.Value("d", 0.0)
    count = manager.Value("i", 0)
    # testSetPrecision = []
    for index, model in enumerate(LSTMModels):
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

