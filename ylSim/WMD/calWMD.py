import sys

sys.path.append('..')

from gensim.models import KeyedVectors
import os
import utils

rootPath = utils.rootPath
modelPath = utils.google_pretrained_path
corpusPath = r'./WMD/total_corpus.txt'
FileNamePath = r'./WMD/RQFileNameIndex'
model = None
requestSentences = {}
serviceSentences = {}

def loadModel():
    global model 
    model = KeyedVectors.load_word2vec_format(path, binary=True)

def getWMD(doc1,doc2):
    
    return model.wmdistance(doc1,doc2)

def loadSentences():
    global serviceSentences
    global requestSentences
    corpus = []
    with open(corpusPath,'r') as f:
        for line in f:
            corpus.append(line.strip().split())
        
    names = []
    with open(FileNamePath,'r') as f:
        for line in f:
            names.append(line.strip()[:-3])
    svCorpus = corpus[:1080]
    rqCorpus = corpus[1080:]
    svNames = names[:1080]
    rqNames = names[1080:]

    for line,name in zip(svCorpus,svNames):
        serviceSentences[name] = line

    for line,name in zip(rqCorpus,rqNames):
        requestSentences[name] = line

def get_topK_relevance(reqName,topK=5):
    if model is None:
        loadSentences()
        loadModel()
    rqSentence = requestSentences[reqName]
    results = []
    for key in serviceSentences:
        results.append((key,getWMD(rqSentence,serviceSentences[key])))

    return sorted(results,key = lambda k : k[1])[:topK]



    