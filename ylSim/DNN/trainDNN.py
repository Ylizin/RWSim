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
import DNN
import DNNLoadData
import utils
from DNN import DNN

_CUDA = torch.cuda.is_available()


def calculateTPFP(pred, r):
    # torch.max will return the indexes of the maxElement at the [1]
    Ones = torch.full_like(r, 1)
    Zeros = torch.full_like(r, 0)

    # for binary classification
    indexPred = torch.max(pred, dim=1)[1].view(-1)  # batch
    # to convert them to byteTensor, for the following calculation
    classified_True = (indexPred == Ones)
    label_True = (r == Ones)
    label_False = (r == Zeros)
    tp = (classified_True & label_True).sum()  # 'and' bit operator
    fp = (classified_True & label_False).sum()
    return tp.item(), fp.item()

    # for the multi classification, we need procedures below
    # topTensor = torch.full_like(r, -1)
    # predTensor = torch.full_like(maxPred, -2)
    # nonTopTensor = torch.full_like(r, 3)

    # # convert true(high relevance) to 3 and non-highrelevance to -1
    # topR = torch.where(r > 2.999, r, topTensor)
    # # convert positive(high relevance) to 3 and non-higpositive to -2
    # topPred = torch.where(maxPred > 2.99, maxPred, predTensor)
    # # convert false(non-highrelevance) to 3 and high-relevance to -1
    # nonTopR = torch.where(r <= 2.99, nonTopTensor, topTensor)

    # tp = 0
    # fp = 0
    # tp = (topR == topPred).cpu().sum()  # tp , predict to be positive in topR
    # # fp , predict to be positive not in topR
    # fp = (topPred == nonTopR).cpu().sum()
    # return tp.item(), fp.item()


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


def customizedLoss(pred, r):
    '''
        pred,r are shape of N,1
        do weighted MSELoss
        output = (ri+0.1)*[(predi-ri)^2]
    '''
    pred = pred.view(-1)
    r = r.view(-1)
    diff = torch.add(pred, -1, r)  # do pred - r
    weighted = torch.add(r, 0.1)  # do r + 0.1
    pow_diff = torch.pow(diff, 2)  # do diff^2
    return torch.mean(torch.mul(pow_diff, weighted))


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


def main():

    parser = argparse.ArgumentParser("DNN")
    parser.add_argument('--outDim', type=int, default=1)
    parser.add_argument('--seqLen', type=int, default=8)
    parser.add_argument('--hiddenDim1', type=int, default=20)
    parser.add_argument('--hiddenDim2', type=int, default=60)
    parser.add_argument('--hiddenDim3', type=int, default=10)

    parser.add_argument('--numWorkers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--foldNum', type=int, default=5)
    parser.add_argument('--level', type=int, default=3)

    parser.add_argument('--nepoch', type=int, default=1000)
    parser.add_argument('--testEvery', type=int, default=10)
    parser.add_argument('--batchSize', type=int, default=128)
    parser.add_argument('--modelFile', default='./models/dnn')

    args = parser.parse_args()

    DNNLoadData.loadFeatures(utils.featurePath)

    train_test_Seqs = DNNLoadData.generateTrainAndTest(args.foldNum)
    # level has 1,2,3 each level we train foldNum models
    DNNModels = [DNN(args) for i in range(args.foldNum)]
    level = args.level

    manager = Manager()
    p = Pool(int(os.cpu_count()/2))
    lock = manager.Lock()
    precision1 = manager.Value('d', 0.0)
    precision2 = manager.Value('d', 0.0)
    precision3 = manager.Value('d', 0.0)
    NDCG = manager.Value('d', 0.0)
    count = manager.Value('i', 0)
    # testSetPrecision = []
    for index, model in enumerate(DNNModels):
        # get the index fold train and test seqs
        ttSeq = train_test_Seqs[index]
        trainSeqs_keys, testSeqs_keys = ttSeq
        trainSeqs = DNNLoadData.getSeqsFromKeys(trainSeqs_keys)
        testSeqs = DNNLoadData.getSeqsFromKeys(testSeqs_keys)
        p.apply_async(trainOneModel, args=(args, model, trainSeqs, testSeqs, testSeqs_keys,
                                           index, count, precision1, precision2, precision3, NDCG, lock), error_callback=utils.errorCallBack)
        # precision = trainOneModel(args,model,trainSeqs,testSeqs,level,index)
        # testSetPrecision.append(precision)
    p.close()
    p.join()
    count = count.value
    precision1 = precision1.value/count
    precision2 = precision2.value/count
    precision3 = precision3.value/count
    NDCG = NDCG.value/count
    print(str(args.foldNum)+'foldCV precision1:' +
          str(precision1))  # +str(np.mean(testSetPrecision)))
    print('precision2 :{}'.format(precision2))
    print('precision3 :{}'.format(precision3))
    print('NDCG : {}'.format(NDCG))
    # trainDataset = DNNLoadData.SimDataSet(trainSeqs1,level=3)
    # testDataset = DNNLoadData.SimDataSet(testSeqs1,level=3)

    # trainDataLoader = DataLoader(
    #     trainDataset, args.batchSize, num_workers=args.numWorkers)
    # testDataloader = DataLoader(
    #     testDataset, args.batchSize, num_workers=args.numWorkers)

    # model = DNN(args)
    # # 1, 5, 5, 5 is a nice weight for rrelu
    # lossWeight = torch.tensor([10.0, 50])
    # if _CUDA:
    #     torch.cuda.set_device(0)
    #     model = model.cuda()
    #     # default GPU is 0
    #     lossWeight = lossWeight.cuda()

    # # add weight to emphasize the high relevance case
    # lossFunc = nn.CrossEntropyLoss(lossWeight)
    # optimizer = optim.Adam(model.parameters(), args.lr)
    # scheduler = StepLR(optimizer, step_size=60, gamma=0.5)

    # bestPrecision = 0.0
    # for i in range(args.nepoch):
    #     tp = 0.
    #     fp = 1.

    #     totalLoss = 0.0
    #     scheduler.step()

    #     model.train()
    #     for seq, r in trainDataLoader:
    #         if _CUDA:
    #             seq = seq.cuda()
    #             r = r.cuda()
    #         r.view(-1)
    #         pred = model(seq)

    #         l = lossFunc(pred, r)
    #         totalLoss += l.item()
    #         _tp, _fp = calculateTPFP(pred, r)

    #         tp += _tp
    #         fp += _fp
    #         optimizer.zero_grad()
    #         l.backward()
    #         optimizer.step()
    #     _p = (tp/(tp+fp))
    #     print('tp:{},fp:{}'.format(tp, fp))
    #     print('epoch:{},Training loss :{:.4},Precision:{:.4}'.format(i, totalLoss, _p))

    #     if i % args.testEvery == (args.testEvery - 1):
    #         tp = 0
    #         fp = 1
    #         model.eval()
    #         for seq, r in testDataloader:
    #             if _CUDA:
    #                 seq = seq.cuda()
    #                 r = r.cuda()
    #             r.view(-1)
    #             pred = model(seq)
    #             _tp, _fp = calculateTPFP(pred, r)
    #             tp += _tp
    #             fp += _fp

    #         precision = (tp / (tp+fp))
    #         if precision > bestPrecision:
    #             torch.save(model.state_dict(), args.modelFile)
    #             bestPrecision = precision
    #         print('tp:{}    fp:{}'.format(tp, fp))
    #         print('epoch:{},Precision:{:.4}'.format(i, precision))
    # print('bestPrecision:{}'.format(bestPrecision))


if __name__ == '__main__':
    utils.generateDirs('./models')
    main()
