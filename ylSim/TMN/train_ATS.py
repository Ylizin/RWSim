import argparse
import os


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader


from .TMNLoadData import (
    NTMDataLoader,
    NTMDataSet,
    loadFeatures,
    generateTrainAndTest,
    getSeqsFromKeys,
    calculateLevelsPN,
    customizedLoss2
)
from .NTMModel import _CUDA, cos, mse
from .trainNTMModel import load_model
from .att_TSModel import ATTSModel


def trainATS(args, model, train_keys, test_keys,index = 0):
    train_seqs = getSeqsFromKeys(train_keys)
    test_seqs = getSeqsFromKeys(test_keys)
    data_set = NTMDataSet(train_seqs)
    data_loader = NTMDataLoader(data_set)

    if _CUDA:
        torch.cuda.set_device(0)
        model = model.cuda()

    loss_func = customizedLoss2
    optimizer = optim.Adam(model.fine_tune_parameters(), args.lr, weight_decay=1e-5)

    bestPrecision = 0.0
    bestNDCG = 0.0
    for i in range(args.nepoch):
        totalLoss = 0.0
        model.train()
        for req_b, req, wsdl_b, wsdl, rel in data_loader:
            r = torch.tensor(rel)
            dist = model(req_b, wsdl_b)
            if _CUDA:
                r = r.cuda()
            r = r.view(-1)
            r = r.type_as(dist)
            vae_loss = loss_func(dist,r)
            l = loss_func(dist, r)
            l = l + vae_loss
            totalLoss += l.item()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        print("epoch:{},Training loss :{:.4}".format(i, totalLoss))

        if i % 20 == 20-1:
            precision1 = 0.0
            precision2 = 0.0
            precision3 = 0.0
            NDCG = 0.0
            model.eval()
            for key in train_keys:  # do evaluation for every key respectively
                predicts = []
                evalSeqs = getSeqsFromKeys(key)
                evalDataSet = NTMDataSet(evalSeqs)
                evalDataloader = NTMDataLoader(evalDataSet)
                for req_b, req, wsdl_b, wsdl, rel in evalDataloader:
                    r = torch.tensor(rel)
                    if _CUDA:
                        r = r.cuda()
                    r = r.view(-1)
                    pred = model(req_b, wsdl_b)
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

            precision1 = precision1 / len(train_keys)
            precision2 = precision2 / len(train_keys)
            precision3 = precision3 / len(train_keys)
            NDCG = NDCG / len(train_keys)
            NDCG = NDCG.item()
            print(
                "epoch:{},Precision1:{:.4},Precision2:{:.4},Precision3:{:.4},NDCG:{:.4}".format(
                    i, precision1, precision2, precision3, NDCG
                )
            )
            if NDCG > bestNDCG or bestNDCG == 0.0:
                torch.save(model.state_dict(), args.modelFile + r'.ATS')
                bestPrecision = precision1
                bestNDCG = NDCG
                if bestNDCG > 0.920:
                    break
    p1 = 0.0
    p2 = 0.0
    p3 = 0.0
    NDCG = 0.0
    model.eval()

    for key in test_keys:  # do evaluation for every key respectively
        predicts = []
        evalSeqs = getSeqsFromKeys(key)
        evalDataSet = NTMDataSet(evalSeqs)
        evalDataloader = NTMDataLoader(evalDataSet)
        for req_b, req, wsdl_b, wsdl, rel in evalDataloader:
            r = torch.tensor(rel)
            if _CUDA:
                r = r.cuda()
            r = r.view(-1)
            pred = model(req_b, wsdl_b)
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

    precision1 = p1 / len(test_keys)
    precision2 = p2 / len(test_keys)
    precision3 = p3 / len(test_keys)
    NDCG = NDCG / len(test_keys)
    NDCG = NDCG.item()
    print('-p1:{}\n-p2:{}\n-p3:{}\n-NDCG:{}'.format(precision1,precision2,precision3,NDCG))
                



def main():
    parser = argparse.ArgumentParser("VAE")
    parser.add_argument("--vocab_size", type=int, default=646)
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--topic_size", type=int, default=120)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--nepoch", type=int, default=500)
    parser.add_argument("--modelFile", default="./TMN/NTM_l1")
    args = parser.parse_args()

    loadFeatures()
    # train_seqs_keys = generateTrainAndTest(5)
    # train_seqs_keys = train_seqs_keys[0][0]+train_seqs_keys[0][1]
    train_keys, test_keys = generateTrainAndTest(5)[0]
    vae_model = load_model()

    model = ATTSModel(args, vae_model=vae_model)
    trainATS(args, model, train_keys, test_keys)


if __name__ == "__main__":
    main()

