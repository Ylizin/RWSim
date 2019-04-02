import os
import utils

rootPath = utils.rootPath
relevanceDir = rootPath+r"/WsdlLDA/reqRelevance"

relevanceDict = {}

def readRelevanceFile(filePath):
    highRelevance = []
    midRelevance = []
    lowRelevance = []
    nonRelevance = []
    with open(filePath,'r') as f:
        for line in f:
            data = line.strip().split()
            name = data[0]
            relevance = int(data[1])
            if relevance == 0:
                nonRelevance.append(name)
            elif relevance == 1:
                lowRelevance.append(name)
            elif relevance == 2:
                midRelevance.append(name)
            elif relevance ==3:
                highRelevance.append(name)
    return highRelevance,midRelevance,lowRelevance,nonRelevance

def loadRelevance():
    
    if relevanceDict:# if the relevance Dic is already loaded, then do nothing 
        return
    for file in os.listdir(relevanceDir):
        fullpath = os.path.join(relevanceDir,file)
        if os.path.isdir(fullpath):
            continue
        dict = {}
        highRelevance,midRelevance,lowRelevance,nonRelevance = readRelevanceFile(fullpath)
        dict['highRelevance'] = highRelevance
        dict['midRelevance'] = midRelevance
        dict['lowRelevance'] = lowRelevance
        dict['nonRelevance'] = nonRelevance
        relevanceDict[file] = dict
 

if __name__ == '__main__':
    loadRelevance()