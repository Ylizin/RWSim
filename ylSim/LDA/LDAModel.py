import gensim 
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.dictionary import Dictionary as gen_dict


import LDA.LDALoadData as load_data

def make_lda_model():
    lda_text = load_data.load_LDA_data()
    lda_dict = gen_dict(lda_text)
    lda_corpus = [lda_dict.doc2bow(text) for text in lda_text]

    lda_model = LdaMulticore(lda_corpus,num_topics= 120,id2word=lda_dict)
    return lda_model,lda_dict,lda_corpus

if __name__ == '__main__':
    model,di,co = make_lda_model()
    co = co[0]
    # _,vec = zip(*model[co])
    print(model[co])
    print(model.get_document_topics(co,minimum_probability= 1e-6))
