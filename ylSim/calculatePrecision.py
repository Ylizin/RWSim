# -*- coding:utf-8 -*-

# from the paper of Gao Panpan, we only focus on the top 5 predict result

import loadRelevance
import numpy as np

loadRelevance.loadRelevance()
relevanceDict = loadRelevance.relevanceDict


def calculateDCG(rel, K0=1, *, K1):
    '''
    rel : the level here is 3,2,1 \n
    K0 : the start index , included \n
    K1 : the end index , included \n
    DCG = from K0-K1 accumulate 2^rel - 1/(log2 (i+1) ) \n
    if K1 = K0 then it equals to calculate the DCG at index K0 \n
    return: DCG
    '''
    DCG = 0.0
    for i in range(K0, K1+1):
        DCG += (2**rel -1) / np.log2(i+1)
    return DCG


def calculateIDCG(thisReqName, topK):
    thisDict = relevanceDict[thisReqName]
    high_relevant = len(thisDict['highRelevance'])
    mid_relevant = len(thisDict['midRelevance'])
    low_relevant = len(thisDict['lowRelevance'])

    IDCG = 0.0
    if topK <= high_relevant:
        IDCG += calculateDCG(3, K1=topK)
    elif topK <= (high_relevant+mid_relevant):
        IDCG += calculateDCG(3, K1=high_relevant)
        IDCG += calculateDCG(2, high_relevant+1, K1=topK)
    elif topK <= (high_relevant+mid_relevant+low_relevant):
        IDCG += calculateDCG(3, K1=high_relevant)
        IDCG += calculateDCG(2, high_relevant+1,
                             K1=high_relevant + mid_relevant)
        IDCG += calculateDCG(1, high_relevant+mid_relevant+1, K1=topK)
    else:
        IDCG += calculateDCG(3, K1=high_relevant)
        IDCG += calculateDCG(2, high_relevant+1,
                             K1=high_relevant + mid_relevant)
        IDCG += calculateDCG(1, high_relevant+mid_relevant+1,
                             K1=high_relevant+mid_relevant+low_relevant)

    return IDCG


def calculateNDCG(thisReqName, topKPredict, topK):
    thisDict = relevanceDict[thisReqName]
    high_relevance = thisDict['highRelevance']
    mid_relevance = thisDict['midRelevance']
    low_relevance = thisDict['lowRelevance']

    DCG = 0.0
    for i, predict in enumerate(topKPredict):
        rel = 0
        if predict in high_relevance:
            rel = 3
        elif predict in mid_relevance:
            rel = 2
        elif predict in low_relevance:
            rel = 1
        # calculate DCG at position i+1, i started from 0
        DCG += calculateDCG(rel, i+1, K1 = i+1)

    return DCG/calculateIDCG(thisReqName, topK)


def calHighRelevancePrecision(thisReqName, top5Predict, confusionMatrix, topK=5):
    '''
        confusionMatrix : tp fn
                          fp tn
    '''
    thisHighReq = relevanceDict[thisReqName]['highRelevance']
    for predict in top5Predict:
        if predict in thisHighReq:
            # tp
            confusionMatrix[0][0] += 1
        else:
            # fp
            confusionMatrix[1][0] += 1

    return confusionMatrix, calculateNDCG(thisReqName, top5Predict, topK)


def calHighAndMidPrecision(thisReqName, top5Predict, confusionMatrix, topK=5):
    thisHighReq = relevanceDict[thisReqName]['highRelevance']
    thisMidReq = relevanceDict[thisReqName]['midRelevance']
    for predict in top5Predict:
        if predict in thisHighReq or predict in thisMidReq:
            confusionMatrix[0][0] += 1
        else:
            confusionMatrix[1][0] += 1

    return confusionMatrix, calculateNDCG(thisReqName, top5Predict, topK)


def calHighAndMidAndLowPrecision(thisReqName, top5Predict, confusionMatrix, topK=5):
    thisHighReq = relevanceDict[thisReqName]['highRelevance']
    thisMidReq = relevanceDict[thisReqName]['midRelevance']
    thisLowReq = relevanceDict[thisReqName]['lowRelevance']
    for predict in top5Predict:
        if predict in thisHighReq or predict in thisMidReq or predict in thisLowReq:
            confusionMatrix[0][0] += 1
        else:
            confusionMatrix[1][0] += 1

    return confusionMatrix, calculateNDCG(thisReqName, top5Predict, topK)
