import os
import gensim
from gensim.models import doc2vec
import numpy as np

rootPath = os.getcwd()
corpusPath = rootPath + r"\trainDocVec\total_corpus.txt"
DocModel = None
DefaultModelPath = r".\trainDocVec\docVecModel"
DefaultFileNameIndexPath = r'.\trainDocVec\RQFileNameIndex'
tags = None

def generateTaggedDoc(corpus_path,file_name_index_path = DefaultFileNameIndexPath):
    corpus_lines = []
    global tags
    tags = []
    TaggedDocs = []
    with open(corpus_path,'r') as f:
        for line in f:
            corpus_lines.append(line.strip().split())

    with open(file_name_index_path,'r') as f:
        for tag in f:
            tag = tag.strip()
            tags.append(tag)
    
    for line,tag in zip(corpus_lines,tags):
        TaggedDocs.append(doc2vec.TaggedDocument(line,[tag]))# attention , here the tags should be a !!!list!!! 

    return TaggedDocs

def trainDocVec(TaggedDocs, vectorSize=300):
    return doc2vec.Doc2Vec(TaggedDocs, vector_size=vectorSize,epochs=1000)

def saveModel(model, modelPath=DefaultModelPath):
    model.save(modelPath)

def loadModel(modelPath=DefaultModelPath):
    model = None
    if not os.path.exists(modelPath) or os.path.isdir(modelPath):
        model = trainDocVec(generateTaggedDoc(corpusPath))
        saveModel(model)
    else:
        #for test 
        # model = trainDocVec(generateTaggedDoc(corpusPath))
        model = doc2vec.Doc2Vec.load(modelPath)
        loadTags()
    return model
    
def loadTags(file_name_index_path = DefaultFileNameIndexPath):
    global tags 
    tags = []
    with open(file_name_index_path,'r') as f:
        for tag in f:
            tag = tag.strip()
            tags.append(tag)

def get_topK_relevance(reqName):
    reqName = reqName + '_rq'
    global DocModel
    if DocModel is None:
        DocModel = loadModel()
    serviceNames = []
    for tag in tags:
        if tag.rfind('_sv') == (len(tag)-3):# end with '_sv'
            serviceNames.append(tag)
    
    results = []
    for sv in serviceNames:
        results.append((sv[:-3],DocModel.docvecs.similarity(reqName,sv)))
    
    return sorted(results,key = lambda k: k[1],reverse = True)




# def test():
#     global DocModel
#     if DocModel is None:
#         DocModel = loadModel()
#     DocVocab = DocModel.vocabulary
#     DocVecs = DocModel.docvecs

#     sv_vecs = DocVecs.vectors_docs[:1080]
#     rq_vecs = DocVecs.vectors_docs[1080:]
#     sv_tags = tags[:1080]
#     rq_tags = tags[1080:]

#     # print(DocModel.infer_vector(sentence))
#     # print(DocModel['1personbicyclecar_price_service.wsdl_sv'])
#     print(DocVecs.similarity('book_readerreview_service.wsdl_sv','book_readerreview_service.wsdl_rq'))
#     print(DocVecs.most_similar([DocModel['1personbicyclecar_price_service.wsdl_sv']],topn=10))
#     print(DocVecs.distances(['1personbicyclecar_price_service.wsdl_sv'],rq_tags))
#     # print(DocVecs.distance(31,1))
#     # print(DocVecs.vectors_docs.shape)
    
#     saveModel(DocModel)

