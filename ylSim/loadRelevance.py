import os
import utils

rootPath = utils.rootPath
relevanceDir = rootPath+r"\WsdlLDA\reqRelevance"

relevanceDict = {}

def readRelevanceFile(filePath):
    highRelevance = []
    midRelevance = []
    lowRelevance = []
    with open(filePath,'r') as f:
        for line in f:
            data = line.strip().split()
            name = data[0]
            relevance = int(data[1])

            if relevance == 1:
                lowRelevance.append(name)
            elif relevance == 2:
                midRelevance.append(name)
            elif relevance ==3:
                highRelevance.append(name)
    
    return highRelevance,midRelevance,lowRelevance

def loadRelevance():
    global relevanceDict
    for file in os.listdir(relevanceDir):
        fullpath = os.path.join(relevanceDir,file)
        if os.path.isdir(fullpath):
            continue
        dict = {}
        highRelevance,midRelevance,lowRelevance = readRelevanceFile(fullpath)
        dict['highRelevance'] = highRelevance
        dict['midRelevance'] = midRelevance
        dict['lowRelevance'] = lowRelevance
        relevanceDict[file] = dict

if __name__ == '__main__':
    loadRelevance()