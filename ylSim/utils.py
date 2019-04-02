import os
import traceback

rootPath = os.getcwd()
featurePath = rootPath+r'/WsdlLDA/reqRelevance/features'
WSDLPath = rootPath+r'/WsdlLDA/originaWSDLsWords'
RQPath = rootPath+r'/WsdlLDA/originRequestsWords'
RelevancePath = rootPath+r'/WsdlLDA/reqRelevance'
registered_path = rootPath+r'/WsdlLDA/originaWSDLsWords/registered'

RQ_TF_path = rootPath + r'/WsdlLDA/originalRequestsWordsTF'
WSDL_TF_path = rootPath + r'/WsdlLDA/originalWSDLsWordsTF'


bertPath = rootPath + r'/bert'
total_corpus_path = rootPath + r'/trainDocVec/total_corpus.txt'
dw = 300

def generateDirs(dirPath):
    if os.path.exists(dirPath):
        return
    else:
        os.makedirs(dirPath)

def errorCallBack(e):
    # print(str(e))
    print(repr(e))

def get_relLevel(relevanceDict,req_name,wsdl_name):
    relDict = relevanceDict[req_name]
    thisHighReq = relDict['highRelevance']
    thisMidReq = relDict['midRelevance']
    thisLowReq = relDict['lowRelevance']
    rel = None 
    if wsdl_name in thisHighReq:
        rel = 3
    elif wsdl_name in thisMidReq:
        rel = 2
    elif wsdl_name in thisLowReq:
        rel = 1
    else:
        rel = 0
    return rel

def iterate_path(path):
    paths = []
    for file in os.listdir(path):
        full_path = os.path.join(path,file)
        if not os.path.isfile(full_path):
            continue
        paths.append(file)
    return paths