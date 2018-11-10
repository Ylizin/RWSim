# -*- coding:utf-8 -*-

#from the paper of Gao Panpan, we only focus on the top 5 predict result

import loadRelevance

loadRelevance.loadRelevance()
relevanceDict = loadRelevance.relevanceDict

def calHighRelevancePrecision(thisReqName,top5Predict,confusionMatrix): 
    '''
        confusionMatrix : tp fn
                          fp tn
    '''
    thisHighReq = relevanceDict[thisReqName]['highRelevance']
    for predict in top5Predict:
        if predict in thisHighReq:
            #tp
            confusionMatrix[0][0] += 1
        else:
            #fp
            confusionMatrix[1][0] += 1

    return confusionMatrix

def calHighAndMidPrecision(thisReqName,top5Predict,confusionMatrix):
    thisHighReq = relevanceDict[thisReqName]['highRelevance']
    thisMidReq = relevanceDict[thisReqName]['midRelevance']
    for predict in top5Predict:
        if predict in thisHighReq or predict in thisMidReq:
            confusionMatrix[0][0] += 1
        else:
            confusionMatrix[1][0] +=1

    return confusionMatrix

def calHighAndMidAndLowPrecision(thisReqName,top5Predict,confusionMatrix):
    thisHighReq = relevanceDict[thisReqName]['highRelevance']
    thisMidReq = relevanceDict[thisReqName]['midRelevance']
    thisLowReq = relevanceDict[thisReqName]['lowRelevance']
    for predict in top5Predict:
        if predict in thisHighReq or predict in thisMidReq or predict in thisLowReq:
            confusionMatrix[0][0] += 1
        else:
            confusionMatrix[1][0] +=1

    return confusionMatrix