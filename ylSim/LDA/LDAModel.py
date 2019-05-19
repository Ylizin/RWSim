import gensim 
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.dictionary import Dictionary as gen_dict
import os

import LDA.LDALoadData as load_data

__default_model_path = r'./LDA/LDA.model'
__default_dict_path = r'./LDA/LDA.dict'

def __make_lda_model(model_path = __default_model_path,dict_path = __default_dict_path):

    lda_text = load_data.load_LDA_data()
    lda_dict = __load_dict(dict_path)
    if lda_dict is None:
        lda_dict = gen_dict(lda_text)
        __save_dict(lda_dict,dict_path) 

    lda_corpus = [lda_dict.doc2bow(text) for text in lda_text]

    lda_model = __load_model(model_path)
    if lda_model is None:
        lda_model = LdaMulticore(lda_corpus,num_topics= 120,id2word=lda_dict,iterations= 3000)
        __save_model(lda_model,model_path) 


    return lda_model,lda_dict,lda_corpus

def __save_dict(dic,dict_path):
    dic.save(dict_path)

def __load_dict(dict_path):
    dic = None
    if os.path.exists(dict_path):
        dic = gen_dict.load(dict_path,mmap='r')
    return dic

def __save_model(model,model_path):
    model.save(model_path)

def __load_model(model_path):
    model = None
    if os.path.exists(model_path):
        model = LdaMulticore.load(model_path,mmap='r')
    return model

def __get_similarity(model,co1,co2):
    pass

if __name__ == '__main__':
    model,dic,co = __make_lda_model()
    
    for c in co:
        idx,vec = zip(*model[c])
        print(idx)
        print(vec)
    print(model.get_topics().shape)
    print(model.get_term_topics(dic.token2id['car']))
    # the whole possibility vec should be exracted in this way, but there are a lot of zero-equaled value
    print(model.get_document_topics(co,minimum_probability= 1e-6))
