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

import DNN
import DNNLoadData
import utils
from DNN import DNN

_CUDA = torch.cuda.is_available()

def errorCB(e):
    print('error:'+str(e))
    traceback.print_exc()

def calculateTPFP(pred, r):
    # torch.max will return the indexes of the maxElement at the [1]
    Ones = torch.full_like(r,1)
    Zeros = torch.full_like(r,0)

    #for binary classification 
    indexPred = torch.max(pred, dim=1)[1].view(-1) #batch 
    classified_True = (indexPred==Ones)#to convert them to byteTensor, for the following calculation 
    label_True = (r == Ones)
    label_False = (r == Zeros)
    tp = (classified_True & label_True).sum()# 'and' bit operator 
    fp = (classified_True & label_False).sum()
    return tp.item(),fp.item()

    #for the multi classification, we need procedures below
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

def trainOneModel(args,model,trainSeqs,testSeqs,level,index,syncPrecision,lock,doPrint=False):
    if index%5==0:
        doPrint = True
    trainDataset = DNNLoadData.SimDataSet(trainSeqs,level)
    testDataset = DNNLoadData.SimDataSet(testSeqs,level)

    trainDataLoader = DataLoader(
        trainDataset, args.batchSize, num_workers=args.numWorkers)
    testDataloader = DataLoader(
        testDataset, args.batchSize, num_workers=args.numWorkers)
    
    # 1, 5, 5, 5 is a nice weight for rrelu
    lossWeight = torch.tensor([150.0, 100])
    if _CUDA:
        torch.cuda.set_device(0)
        model = model.cuda()
        # default GPU is 0
        lossWeight = lossWeight.cuda()

    # add weight to emphasize the high relevance case
    lossFunc = nn.CrossEntropyLoss(lossWeight)
    optimizer = optim.Adam(model.parameters(), args.lr,weight_decay= 3e-5)
    scheduler = StepLR(optimizer, step_size=60, gamma=0.5)
    
    bestPrecision = 0.0
    for i in range(args.nepoch):
        tp = 0.
        fp = 1.

        totalLoss = 0.0
        scheduler.step()

        model.train()
        #attention! here if u are on Windows, the --numWorker should not be too large otherwise it will overconsume the memory
        for seq, r in trainDataLoader:
         
            if _CUDA:
                seq = seq.cuda()
                r = r.cuda()
            r.view(-1)
            pred = model(seq)

            l = lossFunc(pred, r)
            totalLoss += l.item()
            _tp, _fp = calculateTPFP(pred, r)

            tp += _tp
            fp += _fp
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
        _p = (tp/(tp+fp))
        if doPrint:
            print('tp:{},fp:{}'.format(tp, fp))
            print('epoch:{},Training loss :{:.4},Precision:{:.4}'.format(i, totalLoss, _p))
        if i % args.testEvery == (args.testEvery - 1):
            tp = 0
            fp = 1
            model.eval()
            for seq, r in testDataloader:
                if _CUDA:
                    seq = seq.cuda()
                    r = r.cuda()
                r.view(-1)
                pred = model(seq)
                _tp, _fp = calculateTPFP(pred, r)
                tp += _tp
                fp += _fp

            precision = (tp / (tp+fp))
            if precision > bestPrecision:
                torch.save(model.state_dict(), args.modelFile+str(level)+str(index))
                bestPrecision = precision
            if doPrint:
                print('tp:{}    fp:{}'.format(tp, fp))
                print('epoch:{},Precision:{:.4}'.format(i, precision))
    if doPrint:
        print('bestPrecision:{}'.format(bestPrecision))
    model.load_state_dict(torch.load(args.modelFile+str(level)+str(index)))

    tp = 0
    fp = 1
    model.eval()
    for seq, r in testDataloader:
        if _CUDA:
            seq = seq.cuda()
            r = r.cuda()
        r.view(-1)
        pred = model(seq)
        _tp, _fp = calculateTPFP(pred, r)
        tp += _tp
        fp += _fp
    precision = (tp / (tp+fp))
    print('tp:{}'.format(tp))
    print('fp:{}'.format(fp))
    with lock:
        syncPrecision.value += precision
    # return precision


def main():

    parser = argparse.ArgumentParser("DNN")
    parser.add_argument('--outDim', type=int, default=2)
    parser.add_argument('--seqLen', type=int, default=8)
    parser.add_argument('--hiddenDim1', type=int, default=40)
    parser.add_argument('--hiddenDim2', type=int, default=60)
    parser.add_argument('--hiddenDim3', type=int, default=40)

    parser.add_argument('--numWorkers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--foldNum', type=int, default=5)
    parser.add_argument('--level', type=int, default=1)

    parser.add_argument('--nepoch', type=int, default=500)
    parser.add_argument('--testEvery', type=int, default=10)
    parser.add_argument('--batchSize', type=int, default=128)
    parser.add_argument('--modelFile', default='./models/dnn')

    args = parser.parse_args()

    DNNLoadData.loadFeatures(utils.featurePath)

    ttSeqs = DNNLoadData.generateTrainAndTest(args.foldNum)
    DNNModels = [DNN(args) for i in range(args.foldNum)]#level has 1,2,3 each level we train foldNum models
    level = args.level

    manager = Manager()
    p = Pool(int(os.cpu_count()/2))
    lock = manager.Lock()
    precision = manager.Value('d',0.0)
    # testSetPrecision = []
    for index,model in enumerate(DNNModels):
        ttSeq = ttSeqs[index] #get the index fold train and test seqs
        trainSeqs,testSeqs = ttSeq
        p.apply_async(trainOneModel,args = (args,model,trainSeqs,testSeqs,level,index,precision,lock),error_callback=errorCB)
        # precision = trainOneModel(args,model,trainSeqs,testSeqs,level,index)
        #testSetPrecision.append(precision)
    p.close()
    p.join()

    print(str(level)+'level\t'+str(args.foldNum)+'foldCV precision:' + str(precision.value/args.foldNum))#+str(np.mean(testSetPrecision)))


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
