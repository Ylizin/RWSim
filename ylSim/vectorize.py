import warnings
import utils
warnings.filterwarnings(action="ignore", category=UserWarning, module="gensim")
import gensim
from gensim.models import KeyedVectors

import numpy as np
import os

model = None
dw = utils.dw  # dimension of word vector


def loadWordVector(Path):
    global model
    print("Word Vec loading:\n")
    model = KeyedVectors.load_word2vec_format(Path, binary=True)
    print("Word load complete.\n")
    return model


def word2Vec(Word):
    try:
        return model[Word] #return shape (1,dw)
    except KeyError:
        return None
        # return np.zeros((1, dw))

def IDFWWord2Vec(Word):
    IDFDict =loadIDF('IDF.txt')
    try:
        return IDFDict[Word]*model[Word]
    except KeyError:
         return None
        # return np.zeros((1, dw))


def loadIDF(IDFPath):
    dict = {}
    with open(IDFPath,'r') as f:
        for line in f:
            data = line.strip().split()
            dict[data[0]] = float(data[1])
    
    return dict

def wsdl2VecAVE(Path):
    vecData = None
    with open(Path, "r") as f:
        for line in f:
            data = line.strip().split()
            for word in data:
                wordVec = word2Vec(word)
                if wordVec is None:
                    continue
                if vecData is None:
                    vecData = word2Vec(word).reshape(1, dw)
                else:
                    vecData = np.append(vecData, word2Vec(word).reshape(1, dw), axis=0)

    # print(vecData.shape)
    dirname, filename = os.path.split(Path)
    newpath = os.path.join(dirname, "vec", filename)
    # with open(newpath,'w') as wf:
    IDFW_ave_vec = np.mean(vecData, axis=0)

    np.save(newpath, IDFW_ave_vec)

def wsdl2VecRAW(Path):
    vecData = None
    with open(Path, "r") as f:
        for line in f:
            data = line.strip().split()
            for word in data:
                wordVec = word2Vec(word)
                if wordVec is None:
                    continue
                if vecData is None:
                    vecData = word2Vec(word).reshape(1, dw)
                else:
                    vecData = np.append(vecData, word2Vec(word).reshape(1, dw), axis=0)

    # print(vecData.shape)
    dirname, filename = os.path.split(Path)
    newpath = os.path.join(dirname, "raw_vec", filename)
    # with open(newpath,'w') as wf:
    np.save(newpath, vecData)

def wsdl2VecIDFWeighted(Path):
    vecData = None
    with open(Path, "r") as f:
        for line in f:
            data = line.strip().split()
            for word in data:
                wordVec = IDFWWord2Vec(word)
                if wordVec is None:
                    continue
                if vecData is None:
                    vecData = IDFWWord2Vec(word).reshape(1, dw)
                else:   
                    vecData = np.append(vecData, IDFWWord2Vec(word).reshape(1, dw), axis=0)

    # print(vecData.shape)
    dirname, filename = os.path.split(Path)
    newpath = os.path.join(dirname, "IDFW_ave_vec", filename)
    # with open(newpath,'w') as wf:
    IDFW_ave_vec = np.mean(vecData, axis=0)
    np.save(newpath, IDFW_ave_vec)

def dir2Vec(dirPath):
    for file in os.listdir(dirPath):
        fullpath = os.path.join(dirPath, file)
        if not os.path.isdir(fullpath):
            # wsdl2VecIDFWeighted(fullpath)
            wsdl2VecAVE(fullpath)
            wsdl2VecRAW(fullpath)
            # print(fullpath)


if __name__ == "__main__":
    loadWordVector("./Google.bin.gz")
    dir2Vec("C:\\Users\\dell\\Desktop\\WsdlLDA\\originRequestsWords")
    dir2Vec("C:\\Users\\dell\\Desktop\\WsdlLDA\\originaWSDLsWords")
