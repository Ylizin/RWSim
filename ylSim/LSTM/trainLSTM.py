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


def calculateLevelsPN(key, sortedResult):
    p11, ndcg1 = simplePrecisionNDCG(key, sortedResult, 5, 1, doDCG=True)
    p12, _ = simplePrecisionNDCG(key, sortedResult, 5, 2, doDCG=True)
    p13, _ = simplePrecisionNDCG(key, sortedResult, 5, 3, doDCG=True)
    p21, ndcg2 = simplePrecisionNDCG(key, sortedResult, 10, 1, doDCG=True)
    p22, _ = simplePrecisionNDCG(key, sortedResult, 10, 2, doDCG=True)
    p23, _ = simplePrecisionNDCG(key, sortedResult, 10, 3, doDCG=True)
    p31, ndcg3 = simplePrecisionNDCG(key, sortedResult, 15, 1, doDCG=True)
    p32, _ = simplePrecisionNDCG(key, sortedResult, 15, 2, doDCG=True)
    p33, _ = simplePrecisionNDCG(key, sortedResult, 15, 3, doDCG=True)
    p41, ndcg4 = simplePrecisionNDCG(key, sortedResult, 20, 1, doDCG=True)
    p42, _ = simplePrecisionNDCG(key, sortedResult, 20, 2, doDCG=True)
    p43, _ = simplePrecisionNDCG(key, sortedResult, 20, 3, doDCG=True)

    return (
        (ndcg1 + ndcg2 + ndcg3 + ndcg4) / 4,
        (p11 + p21 + p31 + p41) / 4,
        (p12 + p22 + p32 + p42) / 4,
        (p13 + p23 + p33 + p43) / 4,
    )


def customizedLoss(pred, r):
    """
        pred,r are shape of N,1
        do weighted MSELoss
        output = (ri+0.1)*[(predi-ri)^2]
    """
    pred = pred.view(-1)
    r = r.view(-1)
    diff = torch.add(pred, -1, r)  # do pred - r
    weighted = torch.add(r, 0.1)  # do r + 0.1
    pow_diff = torch.pow(diff, 2)  # do diff^2
    return torch.mean(torch.mul(pow_diff, weighted))


def customizedLoss2(pred, r):
    pred = pred.view(-1)
    r = r.view(-1)
    diff = torch.add(pred, -1, r) * 10  # do pred - r
    pow_diff = torch.pow(diff, 2)
    return torch.mean(pow_diff)


def simplePrecisionNDCG(reqName, pred_r, topK=5, level=3, doDCG=False):
    """
        pred_r is sorted and cut out topK 
        if nr < k then k = nr
    """
    tp = 0
    DCG = 0.0
    IDCG = 1
    len_p = getLen(reqName, level)
    precisionK = len_p if len_p < topK else topK

    for i, t in enumerate(pred_r):
        if i >= topK:
            break
        pred, r = t
        r = r.item()
        if doDCG:
            DCG += calculatePrecision.calculateDCG(r, i + 1, K1=i + 1)
        if r >= level:  # eg.  here we have a r=2 ranked here but level=3
            tp += 1
    if doDCG:
        IDCG = calculatePrecision.calculateIDCG(reqName, topK)
    return tp / (precisionK), DCG / IDCG


def trainOneModel(
    args,
    model,
    trainSeqs,
    testSeqs,
    trainSeqs_keys,
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
# we need test keys for the evaluation metrics
    if index % 5 == 0:
        doPrint = True
    level = args.level
    topK = 5

    trainDataset = LoadData.LSTMDataSet(trainSeqs, eval=False)
    testDataset = LoadData.LSTMDataSet(testSeqs)

    trainDataLoader = LoadData.LSTMDataLoader(trainDataset,batch_size = args.batch_size)
    testDataloader = LoadData.LSTMDataLoader(testDataset,batch_size = args.batch_size)

    # 1, 5, 5, 5 is a nice weight for rrelu
    # lossWeight = torch.tensor([10.0])
    if _CUDA:
        torch.cuda.set_device(0)
        model = model.cuda()
        # default GPU is 0
        # lossWeight = lossWeight.cuda()

    lossFunc = customizedLoss2
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=50, gamma=1.0)

    bestPrecision = 0.0
    bestNDCG = 0.0
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

            r = r.type_as(pred)
            l = lossFunc(pred, r)
            totalLoss += l.item()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        if doPrint:
            print("epoch:{},Training loss :{:.4}".format(i, totalLoss))
        if totalLoss < 1.05e+3:
            break
        if i % args.testEvery == (args.testEvery - 1):
            precision1 = 0.0
            precision2 = 0.0
            precision3 = 0.0
            NDCG = 0.0
            model.eval()

            for key in trainSeqs_keys:  # do evaluation for every key respectively
                predicts = []
                evalSeqs = LoadData.getSeqsFromKeys(key)
                evalDataSet = LoadData.LSTMDataSet(evalSeqs)
                evalDataloader = LoadData.LSTMDataLoader(evalDataSet,batch_size = args.batch_size)
                for seq1, seq2, r in evalDataloader:
                    r = torch.tensor(r)
                    if _CUDA:
                        r = r.cuda()
                    r = r.view(-1)
                    pred = model(seq1, seq2)
                    # pred = nn.functional.softmax(pred, dim=1)
                    # prob, predIndex_long = torch.max(pred, dim=1)
                    # predIndex = predIndex_long.type_as(prob)
                    # pred = torch.add(predIndex, prob)
                    r = r.type_as(pred)
                    # sort by pred , calculate by r
                    predicts += list(zip(pred, r))  # list of (predict,r)
                sortedResult = sorted(predicts, key=lambda k: k[0], reverse=True)
                NDCGs, p1s, p2s, p3s = calculateLevelsPN(key, sortedResult)
                precision1 += p1s
                precision2 += p2s
                precision3 += p3s
                NDCG += NDCGs

            precision1 = precision1 / len(trainSeqs_keys)
            precision2 = precision2 / len(trainSeqs_keys)
            precision3 = precision3 / len(trainSeqs_keys)
            NDCG = NDCG / len(trainSeqs_keys)
            NDCG = NDCG.item()
            
            if doPrint:
                print(
                    "epoch:{},Precision1:{:.4},Precision2:{:.4},Precision3:{:.4},NDCG:{:.4}".format(
                        i, precision1, precision2, precision3, NDCG
                    )
                )
            if NDCG > bestNDCG or bestNDCG == 0.0:
                # if precision1 > bestPrecision or bestPrecision == 0.0 :
                torch.save(model.state_dict(), args.modelFile + str(level) + str(index))
                bestPrecision = precision1
                bestNDCG = NDCG
                # if bestPrecision > 0.970:
                    # break

    if doPrint:
        print("bestPrecision:{}".format(bestPrecision))
        print("bestNDCG:{}".format(bestNDCG))

    model.load_state_dict(torch.load(args.modelFile + str(level) + str(index)))

    p1 = 0.0
    p2 = 0.0
    p3 = 0.0
    NDCG = 0.0
    model.eval()

    for key in testSeqs_keys:  # do evaluation for every key respectively
        predicts = []
        evalSeqs = LoadData.getSeqsFromKeys(key)
        evalDataSet = LoadData.LSTMDataSet(evalSeqs)
        evalDataloader = LoadData.LSTMDataLoader(evalDataSet,batch_size = args.batch_size)
        for seq1, seq2, r in evalDataloader:
            r = torch.tensor(r)
            if _CUDA:
                r = r.cuda()
            r = r.view(-1)
            pred = model(seq1, seq2)
            # pred = nn.functional.softmax(pred, dim=1)
            # prob, predIndex_long = torch.max(pred, dim=1)
            # predIndex = predIndex_long.type_as(prob)
            # pred = torch.add(predIndex, prob)
            r = r.type_as(pred)
            # sort by pred , calculate by r
            predicts += list(zip(pred, r))  # list of (predict,r)
        sortedResult = sorted(predicts, key=lambda k: k[0], reverse=True)
        NDCGs, p1s, p2s, p3s = calculateLevelsPN(key, sortedResult)
        NDCG += NDCGs
        p1 += p1s
        p2 += p2s
        p3 += p3s

    precision1 = p1 / len(testSeqs_keys)
    precision2 = p2 / len(testSeqs_keys)
    precision3 = p3 / len(testSeqs_keys)
    NDCG = NDCG / len(testSeqs_keys)
    NDCG = NDCG.item()
    with open(args.modelFile + "testSeqs", "a") as f:
        f.write(str(level) + str(index) + ":")
        f.write(str(testSeqs_keys) + "\n")

    with lock:
        syncCount.value += 1
        syncPrecision1.value += precision1
        syncPrecision2.value += precision2
        syncPrecision3.value += precision3
        syncNDCG.value += NDCG
    # return precision


def main():

    parser = argparse.ArgumentParser("LSTM")
    parser.add_argument("--outDim", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=150)
    
    # parser.add_argument('--hiddenDim2', type=int, default=60)
    # parser.add_argument('--hiddenDim3', type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.0)
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

