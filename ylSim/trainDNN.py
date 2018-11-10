import argparse

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


def calculateTPFP(pred, r):
    # torch.max will return the indexes of the maxElement at the [1]
    maxPred = torch.max(pred, dim=1)[1]

    topTensor = torch.full_like(r, -1)
    predTensor = torch.full_like(maxPred, -2)
    nonTopTensor = torch.full_like(r, 3)

    # convert true(high relevance) to 3 and non-highrelevance to -1
    topR = torch.where(r > 2.999, r, topTensor)
    # convert positive(high relevance) to 3 and non-higpositive to -2
    topPred = torch.where(maxPred > 2.99, maxPred, predTensor)
    # convert false(non-highrelevance) to 3 and high-relevance to -1
    nonTopR = torch.where(r <= 2.99, nonTopTensor, topTensor)

    tp = 0
    fp = 0
    tp = (topR == topPred).cpu().sum()  # tp , predict to be positive in topR
    # fp , predict to be positive not in topR
    fp = (topPred == nonTopR).cpu().sum()
    return tp.item(), fp.item()


def main():
    _CUDA = torch.cuda.is_available()

    parser = argparse.ArgumentParser("DNN")
    parser.add_argument('--outDim', type=int, default=4)
    parser.add_argument('--seqLen', type=int, default=8)
    parser.add_argument('--hiddenDim1', type=int, default=15)
    parser.add_argument('--hiddenDim2', type=int, default=30)
    parser.add_argument('--hiddenDim3', type=int, default=20)

    parser.add_argument('--numWorkers', type=int, default=6)
    parser.add_argument('--lr', type=float, default=3e-3)

    parser.add_argument('--nepoch', type=int, default=300)
    parser.add_argument('--testEvery', type=int, default=10)
    parser.add_argument('--batchSize', type=int, default=128)
    parser.add_argument('--modelFile', default='./dnn.pt')

    args = parser.parse_args()

    DNNLoadData.loadFeatures(utils.featurePath)
    # for each fold
    trainSeqs1, trainSeqs2, trainSeqs3, trainSeqs4, trainSeqs5, testSeqs1, testSeqs2, testSeqs3, testSeqs4, testSeqs5 = DNNLoadData.generateTrainAndTest()

    trainDataset = DNNLoadData.SimDataSet(trainSeqs1)
    testDataset = DNNLoadData.SimDataSet(testSeqs1)

    trainDataLoader = DataLoader(
        trainDataset, args.batchSize, num_workers=args.numWorkers)
    testDataloader = DataLoader(
        testDataset, args.batchSize, num_workers=args.numWorkers)

    model = DNN(args)
    # 1, 5, 5, 5 is a nice weight for rrelu
    lossWeight = torch.tensor([1.0, 5, 5, 4.0])
    if _CUDA:
        torch.cuda.set_device(0)
        model = model.cuda()
        # default GPU is 0
        lossWeight = lossWeight.cuda()

    # add weight to emphasize the high relevance case
    lossFunc = nn.CrossEntropyLoss(lossWeight)
    optimizer = optim.Adam(model.parameters(), args.lr)
    scheduler = StepLR(optimizer, step_size=60, gamma=0.5)

    bestPrecision = 0.0
    for i in range(args.nepoch):
        tp = 0.
        fp = 1.

        totalLoss = 0.0
        scheduler.step()

        model.train()
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
                torch.save(model.state_dict(), args.modelFile)
                bestPrecision = precision
            print('tp:{}    fp:{}'.format(tp, fp))
            print('epoch:{},Precision:{:.4}'.format(i, precision))
    print('bestPrecision:{}'.format(bestPrecision))


if __name__ == '__main__':
    main()
