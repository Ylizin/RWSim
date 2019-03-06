import os
import random
import sys

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

import loadRelevance
import utils

sys.path.append('..')

from .LoadData import LSTMDataSet,LSTMDataLoader


bertPath = utils.bertPath

relevanceDict = {}
reqFeatures = {}
wsdlFeatures={}
reqFeaturePath = utils.RQPath
wsdlFeaturePath = utils.WSDLPath

def loadFeatures(relevancePath= utils.RelevancePath, wsdlPath =utils.WSDLPath):
    '''for every req in rel_path, we load it 
    
    Keyword Arguments:
        relevancePath {[type]} -- [description] (default: {utils.RelevancePath})
        wsdlPath {[type]} -- [description] (default: {utils.WSDLPath})
    '''

    loadRelevance.loadRelevance()
    global relevanceDict
    relevanceDict.update(loadRelevance.relevanceDict)

    for file in os.listdir(relevancePath):
        fullpath = os.path.join(relevancePath,file)
        if os.path.isdir(fullpath):
            continue  
        fullpath = os.path.join(reqFeaturePath,file)
        with open(fullpath,'r') as f:
            for line in f:
                line = line.strip()
                reqFeatures[file] = line
    
    for file in os.listdir(wsdlPath):
        fullpath = os.path.join(wsdlPath,file)
        if os.path.isdir(fullpath):
            continue
        with open(fullpath,'r') as f:
            for line in f:
                line = line.strip()
                wsdlFeatures[file] = line
    print('features reading complete')

# def loadFeatures(bert_path = bertPath):
#     '''for every file in bertPath we load it 
    
#     Keyword Arguments:
#         bert_path {[type]} -- [description] (default: {bertPath})
#     '''
#     loadRelevance.loadRelevance()
#     global relevanceDict
#     if not relevanceDict: # if relevance dict has nothing in it
#         relevanceDict.update(loadRelevance.relevanceDict)


    # global reqFeatures
    # features = bert_gen.generate_bert_vecs_forPT()
    # for tup in features:
    #     req,wsdl,req_vec,wsdl_vec = tup
    #     req_vec = req_vec.numpy()
    #     wsdl_vec = wsdl_vec.numpy()
    #     rel = utils.get_relLevel(relevanceDict,req,wsdl)
    #     reqFeatures.get(req,[]).append((req_vec,wsdl_vec,rel))

        
    # for file in os.listdir(bert_path):
    #     features = []
    #     fullpath = os.path.join(bert_path,file)
    #     if os.path.isdir(fullpath):
    #         continue    
    #     with open(fullpath,'r') as f:
    #         for line in f:
    #             wsdl_name,req,wsdl = bert_sim.processData(line)
    #             req = np.array(req)
    #             wsdl = np.array(wsdl)
    #             req_name = file
    #             rel = utils.get_relLevel(relevanceDict,req_name,wsdl_name)
    #             features.append((req,wsdl,rel))
        
    #     reqFeatures[file] = features
    
def generateTrainAndTest(cvNum):
    '''
     do cvNum fold cross validation
     return train , test seqs
    '''
    seqs_keys = list(reqFeatures.keys())

    # random the seqs for each invoke
    random.shuffle(seqs_keys)
    total_len = len(seqs_keys)
    fold_len = int(total_len/cvNum)
    train_testLists = []
    for i in range(1, cvNum+1):
        train_keys = seqs_keys[:(i-1)*fold_len] + seqs_keys[i*fold_len:]
        test_keys = seqs_keys[(i-1)*fold_len:i*fold_len]
        train_testLists.append((train_keys,test_keys))
    return train_testLists

def getSeqsFromKeys(keys):
    '''
       careful that for evaluation metrics procedure, requests should be test separately

    '''
    if len(reqFeatures) == 0:
        loadFeatures()

    if isinstance(keys,str) : #if the param is a single str 
        keys = [keys]
    random.shuffle(keys)
    return_seqs = []
    
    for req in keys:
        for wsdl in wsdlFeatures.keys():
            reqF = reqFeatures[req]
            wsdlF = wsdlFeatures[wsdl]
            rel = 0
            rel = utils.get_relLevel(relevanceDict,req,wsdl)
            return_seqs.append((reqF,wsdlF,rel))

    return return_seqs