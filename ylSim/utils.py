import os
import traceback

rootPath = os.getcwd()
featurePath = rootPath+r'\WsdlLDA\reqRelevance\features'
WSDLPath = rootPath+r'\WsdlLDA\originaWSDLsWords'
RQPath = rootPath+r'\WsdlLDA\originRequestsWords'
RelevancePath = rootPath+r'\WsdlLDA\reqRelevance'
total_corpus_path = rootPath + r'\trainDocVec\total_corpus.txt'
dw = 300

def generateDirs(dirPath):
    if os.path.exists(dirPath):
        return
    else:
        os.makedirs(dirPath)

def errorCallBack(e):
    print(repr(e))
