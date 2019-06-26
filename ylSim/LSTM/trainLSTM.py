import argparse
import os

import traceback
from multiprocessing import Manager, Pool,set_start_method

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
from LSTM.totalmodel import RWLSTMModel,_CUDA



def calculateLevelsPN(key, sortedResult,model_str=None):
    '''[summary]
    
        p11 means top5 in level 1

    '''
    p11, ndcg1,r11 = simplePrecisionNDCG(key, sortedResult, 5, 1, doDCG=True)
    p12, _,r12 = simplePrecisionNDCG(key, sortedResult, 5, 2, doDCG=True)
    p13, _,r13 = simplePrecisionNDCG(key, sortedResult, 5, 3, doDCG=True)
    p21, ndcg2,r21 = simplePrecisionNDCG(key, sortedResult, 10, 1, doDCG=True)
    p22, _,r22 = simplePrecisionNDCG(key, sortedResult, 10, 2, doDCG=True)
    p23, _,r23 = simplePrecisionNDCG(key, sortedResult, 10, 3, doDCG=True)
    p31, ndcg3,r31 = simplePrecisionNDCG(key, sortedResult, 15, 1, doDCG=True)
    p32, _,r32 = simplePrecisionNDCG(key, sortedResult, 15, 2, doDCG=True)
    p33, _,r33 = simplePrecisionNDCG(key, sortedResult, 15, 3, doDCG=True)
    p41, ndcg4,r41 = simplePrecisionNDCG(key, sortedResult, 20, 1, doDCG=True)
    p42, _,r42 = simplePrecisionNDCG(key, sortedResult, 20, 2, doDCG=True)
    p43, _,r43 = simplePrecisionNDCG(key, sortedResult, 20, 3, doDCG=True)
    
    out_path = utils.output_result_path

    def cal_f1(p,r):
        if p+r == 0:
            return str(0)
        return str(2*(p*r)/(p+r))
    #files store topk result, with level 1-3
    if model_str:
        with open(os.path.join(out_path,model_str+'_top5_precision.txt'),'a') as f:
            f.write(str(p11)+'\t'+str(p12)+'\t'+str(p13)+'\n')
        with open(os.path.join(out_path,model_str+'_top10_precision.txt'),'a') as f:
            f.write(str(p21)+'\t'+str(p22)+'\t'+str(p23)+'\n')
        with open(os.path.join(out_path,model_str+'_top15_precision.txt'),'a') as f:
            f.write(str(p31)+'\t'+str(p32)+'\t'+str(p33)+'\n')
        with open(os.path.join(out_path,model_str+'_top20_precision.txt'),'a') as f:
            f.write(str(p41)+'\t'+str(p42)+'\t'+str(p43)+'\n')
            
        with open(os.path.join(out_path,model_str+'_top5_recall.txt'),'a') as f:
            f.write(str(r11)+'\t'+str(r12)+'\t'+str(r13)+'\n')
        with open(os.path.join(out_path,model_str+'_top10_recall.txt'),'a') as f:
            f.write(str(r21)+'\t'+str(r22)+'\t'+str(r23)+'\n')
        with open(os.path.join(out_path,model_str+'_top15_recall.txt'),'a') as f:
            f.write(str(r31)+'\t'+str(r32)+'\t'+str(r33)+'\n')
        with open(os.path.join(out_path,model_str+'_top20_recall.txt'),'a') as f:
            f.write(str(r41)+'\t'+str(r42)+'\t'+str(r43)+'\n')

        with open(os.path.join(out_path,model_str+'_top5_f1.txt'),'a') as f:
            f.write(cal_f1(p11,r11)+'\t'+cal_f1(p12,r12)+'\t'+cal_f1(p13,r13)+'\n')
        with open(os.path.join(out_path,model_str+'_top10_f1.txt'),'a') as f:
            f.write(cal_f1(p21,r21)+'\t'+cal_f1(p22,r22)+'\t'+cal_f1(p23,r23)+'\n')
        with open(os.path.join(out_path,model_str+'_top15_f1.txt'),'a') as f:
            f.write(cal_f1(p31,r31)+'\t'+cal_f1(p32,r32)+'\t'+cal_f1(p33,r33)+'\n')
        with open(os.path.join(out_path,model_str+'_top20_f1.txt'),'a') as f:
            f.write(cal_f1(p41,r41)+'\t'+cal_f1(p42,r42)+'\t'+cal_f1(p43,r43)+'\n')

        with open(os.path.join(out_path,model_str+'_top5_ndcg.txt'),'a') as f:
            f.write(str(ndcg1)+'\n')
        with open(os.path.join(out_path,model_str+'_top10_ndcg.txt'),'a') as f:
            f.write(str(ndcg2)+'\n')
        with open(os.path.join(out_path,model_str+'_top15_ndcg.txt'),'a') as f:
            f.write(str(ndcg3)+'\n')
        with open(os.path.join(out_path,model_str+'_top20_ndcg.txt'),'a') as f:
            f.write(str(ndcg4)+'\n')
                
    # #files store topk result, with top 5-20
    # with open(os.path.join(out_path,model_str,'_level_low.txt')) as f:
    #     f.write(p11+'\t'+p21+'\t'+p31+'\t'+p41+'\n')
    # with open(os.path.join(out_path,model_str,'_level_mid.txt')) as f:
    #     f.write(p12+'\t'+p22+'\t'+p32+'\t'+p42+'\n')
    # with open(os.path.join(out_path,model_str,'_level_high.txt')) as f:
    #     f.write(p13+'\t'+p23+'\t'+p33+'\t'+p43+'\n')

    return (
        (ndcg1 + ndcg2 + ndcg3 + ndcg4) / 4,
        (p11 + p21 + p31 + p41) / 4, # this is the average of top 5-20 of level 1
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
    tp_fn = 0

    for t in pred_r:#calculate tp+fn
        pred, r = t
        if r>=level:
            tp_fn += 1

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
    return tp / (precisionK), DCG / IDCG,tp/tp_fn


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
        # if totalLoss < 1.05e+3:
        #     break
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
                if bestNDCG > 0.930:
                    break

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
        with lock:  
            NDCGs, p1s, p2s, p3s = calculateLevelsPN(key, sortedResult,args.prog)
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
    parser.add_argument("--prog", type=str, default=parser.prog)
    parser.add_argument("--outDim", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=150)
    
    # parser.add_argument('--hiddenDim2', type=int, default=60)
    # parser.add_argument('--hiddenDim3', type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.64)
    parser.add_argument("--bidirectional", type=bool, default=True)
    
    # parser.add_argument('--numWorkers', type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--foldNum", type=int, default=5)
    parser.add_argument("--level", type=int, default=3)

    parser.add_argument("--nepoch", type=int, default=20)
    parser.add_argument("--testEvery", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--modelFile", default="./models/LSTM")

    args = parser.parse_args()

    LoadData.loadFeatures()

    train_test_Seqs = LoadData.generateTrainAndTest(args.foldNum)
    # level has 1,2,3 each level we train foldNum models
    LSTMModels = [RWLSTMModel(args) for i in range(args.foldNum)]
    level = args.level

    set_start_method('spawn')
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

