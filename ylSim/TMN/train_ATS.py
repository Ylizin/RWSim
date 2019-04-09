import argparse
import os

from multiprocessing import Manager, Pool
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

import utils
from .TMNLoadData import (
    NTMDataLoader,
    NTMDataSet,
    loadFeatures,
    generateTrainAndTest,
    getSeqsFromKeys,
    calculateLevelsPN,
    customizedLoss2,
)
from .NTMModel import _CUDA, cos, mse
from .trainNTMModel import load_model, _pretrained
from .att_TSModel import ATTSModel


def trainATS(
    args,
    model,
    train_keys,
    test_keys,
    index,
    sync_count,
    sync_precision1,
    sync_precision2,
    sync_precision3,
    sync_NDCG,
    lock,
):
    doPrint = False
    if index % 5 == 0:
        doPrint = True

    train_seqs = getSeqsFromKeys(train_keys, args.pretrained)
    test_seqs = getSeqsFromKeys(test_keys, args.pretrained)
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
            vae_loss = loss_func(dist, r)
            l = loss_func(dist, r)
            l = l + vae_loss
            totalLoss += l.item()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        if doPrint:
            print("epoch:{},Training loss :{:.4}".format(i, totalLoss))
        if totalLoss <1.0e+3:
            break
        if i % 20 == 20 - 1:
            precision1 = 0.0
            precision2 = 0.0
            precision3 = 0.0
            NDCG = 0.0
            model.eval()
            for key in train_keys:  # do evaluation for every key respectively
                predicts = []
                evalSeqs = getSeqsFromKeys(key, args.pretrained)
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
            if doPrint:
                print(
                    "epoch:{},Precision1:{:.4},Precision2:{:.4},Precision3:{:.4},NDCG:{:.4}".format(
                        i, precision1, precision2, precision3, NDCG
                    )
                )
            if NDCG > bestNDCG or bestNDCG == 0.0:
                torch.save(model.state_dict(), args.modelFile +str(index) + r".ATS")
                bestPrecision = precision1
                bestNDCG = NDCG
                # if bestPrecision > 0.990:
                    # break
    p1 = 0.0
    p2 = 0.0
    p3 = 0.0
    NDCG = 0.0
    model.eval()

    for key in test_keys:  # do evaluation for every key respectively
        predicts = []
        evalSeqs = getSeqsFromKeys(key, args.pretrained)
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
    with open(args.modelFile + "testSeqs", "a") as f:
        f.write(str(index) + ":")
        f.write(str(test_keys) + "\n")

    with lock:
        sync_count.value += 1
        sync_precision1.value += precision1
        sync_precision2.value += precision2
        sync_precision3.value += precision3
        sync_NDCG.value += NDCG


def main():
    parser = argparse.ArgumentParser("VAE")
    parser.add_argument("--vocab_size", type=int, default=646)
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--topic_size", type=int, default=120)

    parser.add_argument("--pretrained", type=bool, default=_pretrained)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--foldNum", type=int, default=5)

    parser.add_argument("--nepoch", type=int, default=500)
    parser.add_argument("--modelFile", default="./TMN/NTM_l1")
    args = parser.parse_args()

    loadFeatures()
    # train_seqs_keys = generateTrainAndTest(5)
    # train_seqs_keys = train_seqs_keys[0][0]+train_seqs_keys[0][1]
    train_test_Seqs = generateTrainAndTest(args.foldNum)
    vae_model = load_model()
    ATS_models = [ATTSModel(args, vae_model=vae_model) for i in range(args.foldNum)]

    manager = Manager()
    p = Pool(int(os.cpu_count() / 2))
    lock = manager.Lock()
    precision1 = manager.Value("d", 0.0)
    precision2 = manager.Value("d", 0.0)
    precision3 = manager.Value("d", 0.0)
    NDCG = manager.Value("d", 0.0)
    count = manager.Value("i", 0)

    for index, model in enumerate(ATS_models):
        ttSeq = train_test_Seqs[index]
        train_keys, test_keys = ttSeq
        p.apply_async(
            trainATS,
            args=(
                args,
                model,
                train_keys,
                test_keys,
                index,
                count,
                precision1,
                precision2,
                precision3,
                NDCG,
                lock,
            ),
            error_callback = utils.errorCallBack,
        )
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


if __name__ == "__main__":
    main()

