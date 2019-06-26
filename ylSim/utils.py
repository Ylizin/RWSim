import os
import traceback

rootPath = os.getcwd()
featurePath = rootPath+r'/WsdlLDA/reqRelevance/features'
WSDLPath = rootPath+r'/WsdlLDA/originaWSDLsWords'
RQPath = rootPath+r'/WsdlLDA/originRequestsWords'
RelevancePath = rootPath+r'/WsdlLDA/reqRelevance'
registered_path = rootPath+r'/WsdlLDA/originaWSDLsWords/registered'

RQ_IDX_Path = rootPath + r'/WsdlLDA/originaRQsIDXs'
WSDL_IDX_Path = rootPath + r'/WsdlLDA/originalWSDLsIDXs'
RQ_TF_path = rootPath + r'/WsdlLDA/originalRequestsWordsTF'
WSDL_TF_path = rootPath + r'/WsdlLDA/originalWSDLsWordsTF'

google_pretrained_path = r''
extract_w2v_path = r'./load_pretrained_wv/pret'

bertPath = rootPath + r'/bert'
total_corpus_path = rootPath + r'/trainDocVec/total_corpus.txt'
rq_LDA_path = rootPath + r'/WsdlLDA/rq_LDA'
wsdl_LDA_path = rootPath + r'/WsdlLDA/wsdl_LDA'
dw = 300

OWLS_path = rootPath + r'/WsdlLDA/originaOWLSsWords'
OWLS_query_path = rootPath + r'/WsdlLDA/originOWLSqueryWords'
OWLS_RelevancePath = rootPath+r'/WsdlLDA/OWLSreqRelevance'
OWLS_IDX_Path = rootPath + r'/WsdlLDA/OWLSWSDLsIDXs'
OWLS_RQ_IDX_Path = rootPath + r'/WsdlLDA/OWLSRQsIDXs'
OWLS_RQ_TF_path = rootPath + r'/WsdlLDA/originOWLSqueryWordsTF'
OWLS_WSDL_TF_path = rootPath + r'/WsdlLDA/originalOWLSsWordsTF'
OWLS_rq_LDA_path = rootPath + r'/WsdlLDA/OWLS_query_LDA'
OWLS_wsdl_LDA_path = rootPath + r'/WsdlLDA/OWLS_LDA'


RQPath = OWLS_query_path
RelevancePath = OWLS_RelevancePath
WSDLPath = OWLS_path
RQ_IDX_Path = OWLS_IDX_Path
WSDL_IDX_Path = OWLS_RQ_IDX_Path
RQ_TF_path = OWLS_RQ_TF_path
WSDL_TF_path = OWLS_WSDL_TF_path
rq_LDA_path = OWLS_rq_LDA_path
wsdl_LDA_path = OWLS_wsdl_LDA_path

output_result_path = os.path.dirname(rootPath)+r'/results'

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